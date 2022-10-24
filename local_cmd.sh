MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --save_interval 10000"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 32 --class_cond True --mode train --output_index True --adv_noise_num 5000"

# CUDA_VISIBLE_DEVICES='1,2' python scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

MY_CMD="mpiexec -n 1 python scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS"

# MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

# MY_CMD="python scripts/image_sample.py --model_path /mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/results/64250540_1/ema_0.9999_137500.pt $MODEL_FLAGS $DIFFUSION_FLAGS"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='2' $MY_CMD
