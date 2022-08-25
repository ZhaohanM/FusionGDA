cd ../src/; python pretrain_gda_ml_adapter.py \
    --device 'cuda:1' \
    --lr 0.0001 \
    --use_adapter \
    --weight_decay 0.01 \
    --batch_size 12 \
    --amp \
    --reduction_factor 32 \
    --save_step 10000