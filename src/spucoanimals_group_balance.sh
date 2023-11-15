python src/spucoanimals_group_balance.py --seed=0 --gpu=1 --lr=1e-3 --weight_decay=1e-4 > logs/spucoanimals_group_balance_0.log &

sleep 10

python src/spucoanimals_group_balance.py --seed=1 --gpu=1 --lr=1e-3 --weight_decay=1e-4 > logs/spucoanimals_group_balance_1.log &

sleep 10

python src/spucoanimals_group_balance.py --seed=2 --gpu=1 --lr=1e-3 --weight_decay=1e-4 > logs/spucoanimals_group_balance_2.log &


# # set gpu ids
# gpus=(0 1 2 3 4 5 6 7)

# lr_list=(1e-1 1e-2 1e-3 1e-4)
# weight_decay_list=(1e-4)
# arch_list=("resnet18" "resnet50")

# for arch in ${arch_list[@]}; do

#     for lr in ${lr_list[@]}; do

#         for weight_decay in ${weight_decay_list[@]}; do

#             # while the job is not running on any gpu
#             while ! ps -ef | grep "python src/spucoanimals_group_balance.py --lr ${lr} --weight_decay ${weight_decay} --arch ${arch}" | grep -v grep; do

#                 # find gpus that have at least 15000 MB free memory at the moment after running the job for 10 seconds 
#                 for gpu in ${gpus[@]}; do
#                     if [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk "NR==${gpu}+1" | awk '{print $1}') -gt 15000 ]; then
#                         # try to run the job on this gpu. If it fails, try the next gpu

#                         # if the job is running on this gpu in the background, break the loop
#                         if python src/spucoanimals_group_balance.py --lr ${lr} --weight_decay ${weight_decay} --arch ${arch} --gpu ${gpu} > logs/spucoanimals_group_balance_lr${lr}_wd${weight_decay}_${arch}.log & then

#                             # wait for 10 seconds and try again
#                             sleep 100

#                             # if the job is still running, break the loop
#                             break
#                         fi
#                     fi
#                 done

#                 # if the job is not running on any gpu, wait for 10 seconds and try again
#                 sleep 10
#             done

#         done
#     done
# done