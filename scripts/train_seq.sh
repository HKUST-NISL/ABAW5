python3 main.py --num_workers 8 \
 -gpus 1 --trainer_name eri_seq --load_feature vgg  \
 --snippet_size 40 --batch_size 128 --optimizer adam \
  --lr_scheduler step --num_epochs 200  \
  --lr 1e-4 \
  --pretrained /data/pretrained/model-epoch=07-val_total=1.54.ckpt \
  --data_dir /data/abaw5/ \
  --sample_times 1 --sampling_strategy 0 \
  --num_workers 5