#CUDA_VISIBLE_DEVICES=0 python3 test.py --data recycle.yaml --weights ./weights/yolov5_m_0324.pt

#CUDA_VISIBLE_DEVICES=0 python3 test.py --data recycle.yaml --weights ./runs/sparsity_train/exp15/weights/last.pt

####CUDA_VISIBLE_DEVICES=0 python3 test_prune.py --data recycle.yaml --weights ./weights/pruned.pt

#CUDA_VISIBLE_DEVICES=0 python3 test.py --data recycle.yaml --weights ./runs/train/exp231/weights/last.pt

#0430
CUDA_VISIBLE_DEVICES=0 python3 test.py --data recycle.yaml --weights ./runs/Iterative_Pruning/exp10/weights/last.pt
#CUDA_VISIBLE_DEVICES=0,1 python test.py --weights weights/yolov5_0430.pt --data recycle.yaml --img 1280