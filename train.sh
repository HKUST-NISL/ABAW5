python main.py --snippet_size 10 --batch_size 6 --input_size 224 \
--optimizer adamw --lr_scheduler cosine --num_epochs 50 --lr 1e-4 \
--model_name resnet50 --pretrained pretrained/resnet50_ft_weight.pkl \
--data_dir ~/Data/abaw5_sub