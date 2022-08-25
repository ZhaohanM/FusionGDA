device='cuda:1'
lr=5e-5

for lr in 1e-4 1e-5 5e-5 1e-6;
do
python finetune_dpa.py \
    --device $device \
    --batch_size 320 \
    --lr $lr \
    --freeze_disease \
    --test
done

for lr in 1e-4 1e-5 5e-5 1e-6;
do
python finetune_dpa.py \
    --device $device \
    --batch_size 320 \
    --freeze_prot \
    --freeze_disease \
    --lr $lr
done