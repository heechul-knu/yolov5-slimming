#CUDA_VISIBLE_DEVICES=0 python3 test_prune.py --data recycle.yaml --weights ./weights/yolov5_m_0324.pt

CUDA_VISIBLE_DEVICES=0 python3 test_prune.py --data recycle.yaml --weights ./weights/pruned.pt