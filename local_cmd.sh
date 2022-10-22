MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --save_interval 10000"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

# CUDA_VISIBLE_DEVICES='1,2' python scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

CUDA_VISIBLE_DEVICES='0,1,2' mpiexec -n 3 python scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
