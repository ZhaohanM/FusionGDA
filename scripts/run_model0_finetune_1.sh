device='cuda:0'
lr=5e-5

for lr in 1e-4 1e-5 5e-5 1e-6;
do
python finetune_dpa.py \
    --device $device \
    --batch_size 24 \
    --lr $lr \
    --test
done

for lr in 1e-4 1e-5 5e-5 1e-6;
do
python finetune_dpa.py \
    --device $device \
    --batch_size 24 \
    --freeze_prot \
    --lr $lr
done