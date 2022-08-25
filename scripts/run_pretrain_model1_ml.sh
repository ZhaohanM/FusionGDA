save_path_prefix='../save_model_ckp/model1_ml_pretrain_freeze_prot/'
cd ../src/; python pretrain_model1_ml.py \
    --device 'cuda:0' \
    --save_path_prefix $save_path_prefix \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --batch_size 4 \
    --amp \
    --save_step 100