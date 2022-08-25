lr_list=(0.005)
device='cuda:0'
reduction_factor=8
batch_size=80
gradient_accumulation_steps=12

for lr in ${lr_list[@]}; 
do
    cd ../src/pretrain; python pretrain_dda_ml_adapter.py \
        --device $device \
        --lr $lr \
        --use_adapter \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --loss ms_loss \
        --save_step 100 \
        --warmup_steps 100 \
        --reduction_factor $reduction_factor
done