save_path_prefix='../save_model_ckp/pretrain_freeze_disease_encoder_0610/'
python pretrain_dpa.py \
    --device 'cuda:1' \
    --save_path_prefix $save_path_prefix \
    --freeze_disease_encoder \
    --lr 5e-6 \
    --batch_size 24 \
    --save_step 10000