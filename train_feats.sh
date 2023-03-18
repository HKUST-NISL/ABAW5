# python main.py --snippet_size 0 --batch_size 48 --sample_times 1 \
# --optimizer adam --lr_scheduler cosine --lr 1e-4 --lr_decay_rate 0.8 \
# --num_epochs 10 --num_workers 8 --data_dir ~/Data/abaw5 --features $1 --loss l2

python main.py --snippet_size 0 --batch_size 32 --sample_times 1 \
--optimizer adamw --lr_scheduler step --lr 1e-4 --lr_decay_rate 0.5 --lr_decay_steps 10 \
--num_epochs 100 --num_workers 8 --data_dir dataset/ --features features/ --aligned aligned/ --loss l2 \
-gpus 1

# python main.py --snippet_size 0 --batch_size 48 --sample_times 1 \
# --optimizer adam --lr_scheduler step --lr 2e-5 --lr_decay_rate 0.5 --lr_decay_steps 10\
# --num_epochs 100 --num_workers 8 --data_dir ~/Data/abaw5 --features $1 --loss l2


# python main.py --snippet_size 600 --batch_size 15 --sample_times 1 \
# --optimizer adam --lr_scheduler cosine --lr 2e-5 --lr_decay_rate 0.8 \
# --num_epochs 10 --num_workers 8 --data_dir ~/Data/abaw5 --features $1 --loss l2

