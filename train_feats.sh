python main.py --snippet_size 600 --batch_size 16 --sample_times 1 \
--optimizer adamw --lr_scheduler exponential --lr 4e-5 --lr_decay_rate 0.5 \
--num_epochs 200 --num_workers 8 --data_dir ~/Data/abaw5 --features $1 --loss l2

