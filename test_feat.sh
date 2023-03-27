# python main.py --snippet_size 0 --batch_size 15 --sample_times 1 \
# --optimizer adam --lr_scheduler exponential --lr 2e-5 --lr_decay_rate 0.8 \
# --num_epochs 10 --num_workers 8 --data_dir ~/Data/abaw5 --features $1 --loss l2

python main.py --snippet_size 0 --batch_size 10 --sample_times 1 \
--optimizer adamw --lr_scheduler step --lr 1e-4 \
--num_epochs 100 --num_workers 8 --data_dir ~/Data/abaw5 --features res18_pip --loss l2 \
--test_ckpt /home/yini/ABAW5/experiments/eri/res18_au_377/checkpoints/best-epoch=10-val_apcc=0.377.ckpt  \
--data_dir /data/abaw5/ --features res18_openface_features/ --aligned openface_align/ \
--train False


# python main.py --snippet_size 600 --batch_size 15 --sample_times 1 \
# --optimizer adam --lr_scheduler exponential --lr 2e-5 --lr_decay_rate 0.8 \
# --num_epochs 10 --num_workers 8 --data_dir ~/Data/abaw5 --features $1 --loss l2
