save_path_prefix='../save_model_ckp/model1_pretrain_freeze_prot_encoder_lr002/'
python pretrain_dpa.py \
    --device 'cuda:1' \
    --save_path_prefix $save_path_prefix \
    --freeze_prot_encoder \
    --lr 0.002 \
    --weight_decay 0.01 \
    --batch_size 24 \
    --save_step 10000