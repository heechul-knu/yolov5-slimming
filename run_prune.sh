#python3 prune.py --data recycle.yaml --cfg yolov5m.yaml --weights ./weights/best.pt --batch-size 16

python3 prune.py --data recycle.yaml --cfg yolov5m.yaml --weights ./runs/sparsity_train/exp17/weights/last.pt --batch-size 16 #exp15 #17
