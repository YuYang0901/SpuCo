# start time
# start_time=$(date +%s)

# python src/waterbirds_spare.py --seed=0 --gpu=5 --pretrain --infer_num_epochs=2

# print running time in xx hrs xx mins xx secs
# echo "Running time: $(date -u --date @$(( $(date +%s) - $start_time )) +%H:%M:%S)"

# sleep 10

# python src/waterbirds_spare.py --seed=1 --gpu=6 --pretrain --infer_num_epochs=2 > logs/waterbirds_spare_1.log &

# sleep 10

# python src/waterbirds_spare.py --seed=2 --gpu=7 --pretrain --infer_num_epochs=2 > logs/waterbirds_spare_2.log &


# python src/waterbirds_spare.py --seed=0 --gpu=2 --lr=1e-3 --weight_decay=1e-4 --infer_num_epochs=3 --high_sampling_power=3 > logs/waterbirds_spare_from_scratch_infer_3_3.log &


# sleep 10

# python src/waterbirds_spare.py --seed=1 --gpu=1 --lr=1e-3 --weight_decay=1e-4 --infer_num_epochs=3 --high_sampling_power=3 > logs/waterbirds_spare_from_scratch_infer_3_4.log &

# sleep 10

# python src/waterbirds_spare.py --seed=2 --gpu=0 --lr=1e-3 --weight_decay=1e-4 --infer_num_epochs=3 --high_sampling_power=3 > logs/waterbirds_spare_from_scratch_infer_3_5.log &



# python src/waterbirds_spare.py --seed=0 --gpu=5 --pretrain --infer_num_epochs=2 --high_sampling_power=3 > logs/waterbirds_spare_3.log &

# sleep 10

# python src/waterbirds_spare.py --seed=1 --gpu=6 --pretrain --infer_num_epochs=2 --high_sampling_power=3 > logs/waterbirds_spare_4.log &

# sleep 10

# python src/waterbirds_spare.py --seed=2 --gpu=7 --pretrain --infer_num_epochs=2 --high_sampling_power=3 > logs/waterbirds_spare_5.log &


python src/waterbirds_spare.py --seed=0 --gpu=4 --pretrain --infer_num_epochs=2 --high_sampling_power=2 --cluster_all_classes > logs/waterbirds_spare_cluster_all_classes_0.log &

sleep 10

python src/waterbirds_spare.py --seed=1 --gpu=6 --pretrain --infer_num_epochs=2 --high_sampling_power=2 --cluster_all_classes > logs/waterbirds_spare_cluster_all_classes_1.log &

sleep 10

python src/waterbirds_spare.py --seed=2 --gpu=7 --pretrain --infer_num_epochs=2 --high_sampling_power=2 --cluster_all_classes > logs/waterbirds_spare_cluster_all_classes_2.log &