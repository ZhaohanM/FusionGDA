save_path_prefix='../save_model_ckp/pretrain/'
epoch=2
device='cuda:1'
lr=5e-5

for step in 20 40 60 80 100 120;
do
model_path=$save_path_prefix'epoch_'$epoch'_step_'$step'000.bin'
python finetune_dpa.py \
    --device $device \
    --model_path $model_path \
    --batch_size 24 \
    --lr $lr \
    --test
done

for step in 20 40 60 80 100 120;
do
model_path=$save_path_prefix'epoch_'$epoch'_step_'$step'000.bin'
python finetune_dpa.py \
    --device $device \
    --model_path $model_path \
    --batch_size 24 \
    --lr $lr
done