import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from threading import Thread
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from models.yolo_prune import Model_prune
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    print_mutation, set_logging, one_cycle
from utils.google_utils import attempt_download
from utils.loss import compute_loss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first

import copy
from models.common_prune import *
from models.yolo_prune import *

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")



def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(f'Hyperparameters {hyp}')
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        print(weights)

        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
        #exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
        exclude = []
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=True)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create


    # heechul
    new_cfg(model, nc, ckpt)


# heechul
def new_cfg(model, nc, ckpt):

    #local
    '''
    donntprune = []
    percent = 0.1
    total = 0
    pruned = 0
    cfg = []
    cfg_mask = []
    print('--'*30)
    print("Pre-processing...")

    layer_idx = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            if k not in donntprune:
                size = m.weight.data.shape[0]
                total += m.weight.data.shape[0]
                layer_idx.append(k)
                weight_copy = m.weight.data.abs().clone()
                y, i = torch.sort(weight_copy)
                thre_index = int(size * percent)
                thre = y[thre_index].cuda()
                mask = weight_copy.gt(thre).float().cuda()  # 掩模
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)# 直接修改m，直接改了model的值，并放在了model中
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                #         format(k, mask.shape[0], int(torch.sum(mask))))
            else:
                dontp = m.weight.data.numel()
                mask = torch.ones(m.weight.data.shape)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                #         format(k, dontp, int(dontp)))
                cfg.append(int(dontp))
                cfg_mask.append(mask.clone())
    pruned_ratio = pruned/total
    print(pruned_ratio)
    '''

    #global
    
    for name, module in model.named_modules():
        print(name)


    donntprune = []

    total = 0
    for k,m in enumerate(model.modules()):
         if isinstance(m, nn.BatchNorm2d):
             if k not in donntprune:
                total += m.weight.data.shape[0]

    bn = torch.zeros(total)

    index = 0
    percent = 0.1
    for k,m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            if k not in donntprune:
                size = m.weight.data.shape[0]                
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size
    y, i = torch.sort(bn)# y,i是从小到大排列所有的bn，y是weight，i是序号
    #print(y.shape)

    #print(total)
    thre_index = int(total * percent)
    #print(thre_index)
    thre = y[thre_index].cuda()
    print('threshold:', thre)
    pruned = 0
    cfg = []
    cfg_mask = []
    print('--'*30)
    print("Pre-processing...")

    layer_idx = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            if k not in donntprune:
                layer_idx.append(k)
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()  # 掩模
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                #m.weight.data.mul_(mask)# 直接修改m，直接改了model的值，并放在了model中
                #m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                #         format(k, mask.shape[0], int(torch.sum(mask))))
            else:
                dontp = m.weight.data.numel()
                mask = torch.ones(m.weight.data.shape)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                #         format(k, dontp, int(dontp)))
                cfg.append(int(dontp))
                cfg_mask.append(mask.clone())
    pruned_ratio = pruned/total
    print(pruned_ratio)
    

    
    print('Pre-processing Successful!')
    
    print('--'*30)
    print("\n")
    print("---------------------------------------------------- Making New model ----------------------------------------------------")
    
    newmodel = Model_prune(opt.cfg, ch=3, nc=nc, pc=cfg).to(device)

    #newmodel = model

    for i in range(len(cfg_mask)):
      if newmodel.ch[i] == -1: 
        cfg_mask[i] = torch.ones(cfg_mask[i].shape).float().cuda() #.count_nonzero()

    total_pruned_channel = 0
    total_channel = 0
        
    for i in range(len(cfg_mask)):
      print(newmodel.ch[i], cfg_mask[i].count_nonzero(), cfg_mask[i].shape[0])
      
      if newmodel.ch[i] is not -1:
        total_pruned_channel += newmodel.ch[i]
      else:
        total_pruned_channel += cfg_mask[i].shape[0]
        
      total_channel += cfg_mask[i].shape[0] 
      
    print("pruning ratio: %f" % (total_pruned_channel / total_channel))
    
    
    old_modules = list(model.named_modules())
    new_modules = list(newmodel.named_modules())
    layer_id_in_cfg = 0
    

    bottleneck_last_conv_layer_name = ['model.2.m.1.cv2.conv', 'model.4.m.5.cv2.conv', 'model.6.m.5.cv2.conv', 'model.9.m.1.cv2.conv',  'model.13.m.1.cv2.conv', 'model.17.m.1.cv2.conv','model.20.m.1.cv2.conv', 'model.23.m.1.cv2.conv']
    bottleneck_last_conv_end_mask = []

    for layer_id in range(len(old_modules)):
        name0, m0 = old_modules[layer_id]
        name1, m1 = new_modules[layer_id]

        if name0 in bottleneck_last_conv_layer_name:
            end_mask = cfg_mask[layer_id_in_cfg]
            bottleneck_last_conv_end_mask.append(end_mask.clone())
        elif isinstance(m0, nn.BatchNorm2d):
            layer_id_in_cfg += 1

    layer_id_in_cfg = 0
    start_mask = torch.ones(12)
    end_mask = cfg_mask[layer_id_in_cfg]
    bottleneck_start_mask = 0
    print("pruning...")
    v=0
    concat_start_mask = []
    detect_start_mask = []
    
    conv_layer_name = ['model.1.conv', 'model.3.conv', 'model.5.conv', 'model.7.conv', 'model.10.conv', 'model.14.conv', 'model.18.conv', 'model.21.conv'] # noral conv
    c3_cv1_conv_layer_name = ['model.2.cv1.conv', 'model.4.cv1.conv', 'model.6.cv1.conv', 'model.9.cv1.conv', 'model.13.cv1.conv', 'model.17.cv1.conv', 'model.20.cv1.conv', 'model.23.cv1.conv']
    c3_cv2_conv_layer_name = ['model.2.cv2.conv', 'model.4.cv2.conv', 'model.6.cv2.conv', 'model.9.cv2.conv', 'model.13.cv2.conv', 'model.17.cv2.conv', 'model.20.cv2.conv', 'model.23.cv2.conv']
    c3_cv3_conv_layer_name = ['model.2.cv3.conv', 'model.4.cv3.conv', 'model.6.cv3.conv', 'model.9.cv3.conv', 'model.13.cv3.conv', 'model.17.cv3.conv', 'model.20.cv3.conv', 'model.23.cv3.conv']
    spp_cv1_conv_layer_name = ['model.8.cv1.conv']
    spp_cv2_conv_layer_name = ['model.8.cv2.conv']
    detect_conv_layer_name = ['model.24.m.0', 'model.24.m.1', 'model.24.m.2']
    bottleneck_conv_layer_name = ['model.2.m.0.cv1.conv', 'model.2.m.0.cv2.conv', 'model.2.m.1.cv1.conv', \
        'model.2.m.1.cv2.conv', 'model.4.m.0.cv1.conv', 'model.4.m.1.cv1.conv', 'model.4.m.2.cv1.conv', \
            'model.4.m.3.cv1.conv', 'model.4.m.4.cv1.conv', 'model.4.m.5.cv1.conv', 'model.4.m.0.cv2.conv', \
                'model.4.m.1.cv2.conv', 'model.4.m.2.cv2.conv', 'model.4.m.3.cv2.conv', 'model.4.m.4.cv2.conv', \
                    'model.4.m.5.cv2.conv', 'model.6.m.0.cv1.conv', 'model.6.m.1.cv1.conv', 'model.6.m.2.cv1.conv', \
                        'model.6.m.3.cv1.conv', 'model.6.m.4.cv1.conv', 'model.6.m.5.cv1.conv', 'model.6.m.0.cv2.conv', \
                            'model.6.m.1.cv2.conv', 'model.6.m.2.cv2.conv', 'model.6.m.3.cv2.conv', 'model.6.m.4.cv2.conv', \
                                'model.6.m.5.cv2.conv', 'model.9.m.0.cv1.conv', 'model.9.m.1.cv1.conv', 'model.9.m.0.cv2.conv', \
                                    'model.9.m.1.cv2.conv', 'model.13.m.0.cv1.conv', 'model.13.m.1.cv1.conv', 'model.13.m.0.cv2.conv', \
                                        'model.13.m.1.cv2.conv', 'model.17.m.0.cv1.conv', 'model.17.m.1.cv1.conv', 'model.17.m.0.cv2.conv', \
                                            'model.17.m.1.cv2.conv','model.20.m.0.cv1.conv', 'model.20.m.1.cv1.conv', 'model.20.m.0.cv2.conv', \
                                                'model.20.m.1.cv2.conv', 'model.23.m.0.cv1.conv', 'model.23.m.1.cv1.conv', 'model.23.m.0.cv2.conv', \
                                                    'model.23.m.1.cv2.conv']

#    bn_layer_name = ['bn', 'model.1.bn']


    for layer_id in range(len(old_modules)):
        name0, m0 = old_modules[layer_id]
        name1, m1 = new_modules[layer_id]

        #print(name0)model.0.conv.
        #print(m0)
        if layer_id == 0:
            continue

        if 'model.0.conv.conv' in name0: # Focus
            start_mask = torch.ones(12)
            end_mask = cfg_mask[layer_id_in_cfg]
            print("conv_layer_name")

            ################### weight copy ########################
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('Conv In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            #print(len(idx0.tolist()))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            #######################################################

            # mask for next layer
            start_mask  = end_mask.clone()
            layer_id_in_cfg += 1
        elif name0 in conv_layer_name:
            # normal conv layer
            end_mask = cfg_mask[layer_id_in_cfg]

            ################### weight copy ########################
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('Conv In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            #print(len(idx0.tolist()))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            #######################################################

            # mask for next layer
            start_mask  = end_mask.clone()
            layer_id_in_cfg += 1
        elif name0 in bottleneck_conv_layer_name:
            end_mask = cfg_mask[layer_id_in_cfg]

            ################### weight copy ########################
            idx0 = np.squeeze(np.argwhere(np.asarray(bottleneck_start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('Conv In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            #print(len(idx0.tolist()))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            #######################################################

            # mask for next layer
            bottleneck_start_mask = end_mask.clone()
            layer_id_in_cfg += 1
        elif name0 in c3_cv1_conv_layer_name:
            if 'model.13.cv1.conv' in name0:
                start_mask = torch.Tensor(concat_start_mask[1].tolist()+start_mask.tolist())
            if 'model.17.cv1.conv' in name0:
                start_mask = torch.Tensor(concat_start_mask[0].tolist()+start_mask.tolist())
            if 'model.20.cv1.conv' in name0:
                start_mask = torch.Tensor(concat_start_mask[3].tolist()+start_mask.tolist())
            if 'model.23.cv1.conv' in name0:
                start_mask = torch.Tensor(concat_start_mask[2].tolist()+start_mask.tolist())

            end_mask = cfg_mask[layer_id_in_cfg]

            ################### weight copy ########################
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('Conv In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            #print(len(idx0.tolist()))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            #######################################################

            # mask for next layer
            # start_mask  = end_mask.clone()
            bottleneck_start_mask = end_mask.clone()
            layer_id_in_cfg += 1   
        elif name0 in c3_cv2_conv_layer_name:
            end_mask = cfg_mask[layer_id_in_cfg]

            ################### weight copy ########################
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('Conv In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            #print(len(idx0.tolist()))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            #######################################################

            # mask for next layer
            start_mask  = end_mask.clone()
            layer_id_in_cfg += 1       
        elif name0 in c3_cv3_conv_layer_name:
            bottleneck_start_mask = bottleneck_last_conv_end_mask.pop(0)
            start_mask = torch.Tensor(bottleneck_start_mask.tolist()+start_mask.tolist())

            end_mask = cfg_mask[layer_id_in_cfg]

            ################### weight copy ########################
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('Conv In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            #print(len(idx0.tolist()))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            #######################################################

            # mask for next layer
            start_mask  = end_mask.clone()
            layer_id_in_cfg += 1
        elif name0 in spp_cv1_conv_layer_name:
            end_mask = cfg_mask[layer_id_in_cfg]

            ################### weight copy ########################
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('Conv In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            #print(len(idx0.tolist()))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            #######################################################

            # mask for next layer
            start_mask  = end_mask.clone()
            layer_id_in_cfg += 1            
        elif name0 in spp_cv2_conv_layer_name:
            start_mask = torch.Tensor(start_mask.tolist()+start_mask.tolist()+start_mask.tolist()+start_mask.tolist())
            end_mask = cfg_mask[layer_id_in_cfg]

            ################### weight copy ########################
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('Conv In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            #print(len(idx0.tolist()))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()

            #m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            #######################################################

            # mask for next layer
            start_mask  = end_mask.clone()
            layer_id_in_cfg += 1   

        elif name0 in detect_conv_layer_name:
            start_mask = detect_start_mask.pop(0)

            ################### weight copy ########################
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))

            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            m1.weight.data = w1.clone()

            m1.bias.data = m0.bias.data.clone()
            #######################################################

            # mask for next layer
            layer_id_in_cfg += 1

        if 'model.4.cv3.conv' in name0 or 'model.6.cv3.conv' in name0 or 'model.10.conv' in name0 or 'model.14.conv' in name0:
            concat_start_mask.append(start_mask)

        if 'model.23.cv3.conv' in name0 or 'model.20.cv3.conv' in name0 or 'model.17.cv3.conv' in name0:
            detect_start_mask.append(start_mask)

        if isinstance(m0, nn.BatchNorm2d):# 向新模型中写入
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            #print(m1.weight.data)
            #m1.weight.data[m1.weight.data < 0.5] = 0.
            #m1.bias.data[m1.weight.data < 0.5] = 0.
            #print(np.count_nonzero(m1.weight.data.cpu().numpy()) / m1.weight.data.cpu().numpy().shape[0])
    
         
    #print(model)
    #print(newmodel)

    print('--'*30)
    print('prune done!')
    print('pruned ratio %.3f'%pruned_ratio)
    
    ema = ModelEMA(newmodel) #newmodel #model
    epoch, best_fitness = 0, 0.0

    ckpt['model'] = ema.ema
    ckpt['cfg'] = cfg #cfg #opt.cfg
    
    # Save last, best and delete
    torch.save(ckpt, "./weights/pruned.pt")

    del ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
#    if opt.global_rank in [-1, 0]:
#        check_git_status()

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.global_rank, opt.local_rank = '', ckpt, True, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard

        train(hyp, opt, device, tb_writer, wandb)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, wandb=wandb)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
