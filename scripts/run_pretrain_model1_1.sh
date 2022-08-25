save_path_prefix='../save_model_ckp/model1_pretrain_freeze_disease_encoder_lr002/'
python pretrain_dpa.py \
    --device 'cuda:0' \
    --save_path_prefix $save_path_prefix \
    --freeze_disease_encoder \
    --lr 0.002 \
    --weight_decay 0.01 \
    --batch_size 200 \
    --save_step 10000