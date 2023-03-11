python3 main.py --num_workers 8 -gpus 1 \
--trainer_name eri_single --load_feature vgg   \
--batch_size 500 \
--optimizer adam --lr_scheduler step \
--num_epochs 200 --lr 1e-3 \
--data_dir /data/abaw5/
