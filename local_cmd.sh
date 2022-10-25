MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --save_interval 10000"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --microbatch -1 --class_cond False --mode adv --output_index True --output_class True --learn_sigma False --adv_noise_num 5000"

# CUDA_VISIBLE_DEVICES='1,2' python scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

# torchrun --nnodes=2 --nproc_per_node=8
# mpiexec -n 4

# MY_CMD="mpiexec -n 1 python scripts/image_train.py --data_dir datasets/sub_cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS"

# MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# /home/renjie3/improved-diffusion/results/10/ema_0.9999_012800.pt
# /mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/results/64250540_1/ema_0.9999_137500.pt

MY_CMD="python scripts/image_sample.py --model_path /home/renjie3/improved-diffusion/results/10/ema_0.9999_012800.pt $MODEL_FLAGS $DIFFUSION_FLAGS"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='1' $MY_CMD
