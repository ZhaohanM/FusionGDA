lr_list=(0.00005 0.0005 0.0001 0.001)
device='cuda:0'
reduction_factor=8
batch_size=8
gradient_accumulation_steps=128



for lr in ${lr_list[@]}; 
do
    cd ../src/pretrain/; python pretrain_gda_infoNCE.py \
        --device $device \
        --lr $lr \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --loss infoNCE \
        --use_pooled \
        --save_step 100 \
        --warmup_steps 100 \
        --max_epoch 50 \
        --reduction_factor $reduction_factor
done