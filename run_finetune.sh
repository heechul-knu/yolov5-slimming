CUDA_VISIBLE_DEVICES=0,1 python3 train.py --data recycle.yaml --cfg yolov5m.yaml --weights ./weights/pruned.pt --batch-size 16
