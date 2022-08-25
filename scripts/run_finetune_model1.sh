save_path_prefix='../save_model_ckp/model1_freeze_disease_encoder_0605/'
device='cuda:1'
lr=5e-5

for step in {1..20};
do
model_path=$save_path_prefix'step_'$step'0000_model.bin'
echo $model_path
python finetune_dpa.py \
    --device $device \
    --model_path $model_path \
    --batch_size 24 \
    --lr $lr \
    --test
done

for step in {1..20};
do
model_path=$save_path_prefix'step_'$step'0000_model.bin'
python finetune_dpa.py \
    --device $device \
    --model_path $model_path \
    --batch_size 24 \
    --lr $lr
done