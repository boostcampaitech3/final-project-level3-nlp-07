python train.py  --gradient_clip_val 1.0 --max_epochs 5 --default_root_dir logs  --gpus 1 --batch_size 32 --num_workers 4 --max_len 256
python inference.py
