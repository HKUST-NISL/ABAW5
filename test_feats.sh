# python main.py --snippet_size 0 --batch_size 15 --sample_times 1 \
# --optimizer adam --lr_scheduler exponential --lr 2e-5 --lr_decay_rate 0.8 \
# --num_epochs 10 --num_workers 8 --data_dir ~/Data/abaw5 --features $1 --loss l2

python main.py --snippet_size 0 --batch_size 48 --sample_times 1 \
--optimizer adam --lr_scheduler cosine --lr 2e-5 \
--num_epochs 100 --num_workers 8 --data_dir ~/Data/abaw5 --mode vamm --loss l2 \
--test_ckpt $1 \
--train False


# python main.py --snippet_size 600 --batch_size 15 --sample_times 1 \
# --optimizer adam --lr_scheduler exponential --lr 2e-5 --lr_decay_rate 0.8 \
# --num_epochs 10 --num_workers 8 --data_dir ~/Data/abaw5 --features $1 --loss l2