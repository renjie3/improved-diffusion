#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

MYTIME="47:59:00"
MYNTASKS="1"
MYCPU="4"
MYGRES="gpu:v100:1"

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.3 --learn_sigma False --num_input_channels 1"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--save_interval 5000 --lr 1e-4 --batch_size 100 --microbatch -1 --class_cond False --load_model False --model_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/30/model064000.pt"
ADV_FLAGS="--mode train --output_index True --output_class True --adv_noise_num 5923 --adv_step 30 --save_forward_clean_sample False --single_target_image_id 5000 --adv_loss_type forward_bachword_loss --group_model_dir /egr/research-dselab/renjie3/renjie/improved-diffusion/results/58 --group_model True --group_model_num 6 --random_noise_every_adv_step True --t_seg_num 8 --t_seg_start 3 --eot_gaussian_num 1"
POISON_FLAGS="--poisoned True --poisoned_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/62/adv_noise"
SAMPLE_FLAGS="--poisoned False --sample_starting_from_t False --progress True --poisoned_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/19/adv_noise --in_dir /localscratch/renjie/sub_cifar_train_bird_horse --out_dir /egr/research-dselab/renjie3/renjie/improved-diffusion/results/57/model010000 --t 1000"

JOB_INFO="train 01"
MYCOMMEND="mpiexec -n 1 python -u scripts/image_train.py --data_dir /localscratch/renjie/mnist_train_01 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $ADV_FLAGS $POISON_FLAGS"

MYCOMMEND2="python3 test.py -gpu_id 0 -model 1 -attack 1 --pgd_norm 7 -batch_size 50 -path Final/VANILLA_62162198_1/iter_50 --alpha 1000 --num_iter 100 --num_stop 2000 --test_subset --seed 1"

MYCOMMEND3="python3 test.py -gpu_id 0 -model 1 -attack 1 --pgd_norm 7 -batch_size 50 -path Final/VANILLA_62162198_1/iter_50 --alpha 5000 --num_iter 100 --num_stop 2000 --test_subset --seed 1"

MYCOMMEND2="No_commend2"
MYCOMMEND3="No_commend3"

cat ./slurm_files/sconfigs1_cmse.sb > submit.sb
# cat ./slurm_files/sconfigs1_scavenger.sb > submit.sb
# cat ./slurm_files/sconfigs1.sb > submit.sb
echo "#SBATCH --time=${MYTIME}             # limit of wall clock time - how long the job will run (same as -t)" >> submit.sb
echo "#SBATCH --cpus-per-task=${MYCPU}           # number of CPUs (or cores) per task (same as -c)" >> submit.sb
echo "#SBATCH --ntasks=${MYNTASKS}                  # number of tasks - how many tasks (nodes) that you require (same as -n)" >> submit.sb
echo "#SBATCH --gres=${MYGRES}" >> submit.sb
# echo "#SBATCH --nodelist=nvl-001" >> submit.sb
echo "#SBATCH -o ${MY_JOB_ROOT_PATH}/logfile/%j.log" >> submit.sb
echo "#SBATCH -e ${MY_JOB_ROOT_PATH}/logfile/%j.err" >> submit.sb
cat ./slurm_files/sconfigs2.sb >> submit.sb
echo "JOB_INFO=\"${JOB_INFO}\"" >> submit.sb
echo "MYCOMMEND=\"${MYCOMMEND} --job_id \${SLURM_JOB_ID}_1\"" >> submit.sb
echo "MYCOMMEND2=\"${MYCOMMEND2} --job_id \${SLURM_JOB_ID}_2\"" >> submit.sb
echo "MYCOMMEND3=\"${MYCOMMEND3} --job_id \${SLURM_JOB_ID}_3\"" >> submit.sb
cat ./slurm_files/sconfigs3.sb >> submit.sb
MY_RETURN=`sbatch submit.sb`

echo $MY_RETURN

MY_SLURM_JOB_ID=`echo $MY_RETURN | awk '{print $4}'`

#print the information of a job into one file
date >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MY_SLURM_JOB_ID >>${MY_JOB_ROOT_PATH}/history_job.log
echo $JOB_INFO >>${MY_JOB_ROOT_PATH}/history_job.log
echo "#SBATCH --time=${MYTIME}" >>${MY_JOB_ROOT_PATH}/history_job.log
echo "#SBATCH --cpus-per-task=${MYCPU}" >>${MY_JOB_ROOT_PATH}/history_job.log
echo "#SBATCH --gres=${MYGRES}" >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MYCOMMEND >>${MY_JOB_ROOT_PATH}/history_job.log
if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $MYCOMMEND2 >>${MY_JOB_ROOT_PATH}/history_job.log
fi
if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $MYCOMMEND3 >>${MY_JOB_ROOT_PATH}/history_job.log
fi
echo "---------------------------------------------------------------" >>${MY_JOB_ROOT_PATH}/history_job.log
