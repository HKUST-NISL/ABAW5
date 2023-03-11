# python main.py --snippet_size 10 --batch_size 6 --input_size 224 --sample_times 1 \
# --optimizer adamw --lr_scheduler exponential --num_epochs 100 --lr 1e-3 --lr_decay_rate 0.95 \
# --model_name resnet50 --pretrained pretrained/resnet50_ft_weight.pkl \
# --data_dir ~/Data/abaw5_sub

# python main.py --snippet_size 10 --batch_size 6 --input_size 224 --sample_times 1 \
# --optimizer adamw --lr_scheduler cosine --num_epochs 100 --lr 1e-2 --lr_decay_rate 0.95 \
# --model_name effnetb0 --pretrained '' \
# --data_dir ~/Data/abaw5_sub


python main.py --snippet_size 300 --batch_size 32 --input_size 224 --sample_times 1 \
--optimizer adam --lr_scheduler cosine --num_epochs 200 --lr 1e-4 \
--model_name SMMNet --pretrained pretrained/model-epoch=07-val_total=1.54.ckpt \
--data_dir ~/Data/abaw5 \
--features smm


# python main.py --snippet_size 50 --batch_size 128 --sample_times 1 \
# --optimizer adam --lr_scheduler cosine --num_epochs 50 --lr 1e-4 \
# --data_dir ./dataset/abaw5 \
# --features smm