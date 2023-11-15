# python src/spucomnist_spare.py --seed=0 --gpu=2 --infer_num_epochs=1 --high_sampling_power=1 --feature_difficulty=magnitude_large > logs/spucomnist_spare_power1_0.log & 
# python src/spucomnist_spare.py --seed=1 --gpu=2 --infer_num_epochs=1 --high_sampling_power=1 > logs/spucomnist_spare_power1_1.log &
# python src/spucomnist_spare.py --seed=2 --gpu=2 --infer_num_epochs=1 --high_sampling_power=1 > logs/spucomnist_spare_power1_2.log &

# python src/spucomnist_spare.py --seed=0 --gpu=7 --infer_num_epochs=1 --high_sampling_power=1 --label_noise=0.001 > logs/spucomnist_spare_power1_label_noise_0.001_0.log & 
# python src/spucomnist_spare.py --seed=1 --gpu=6 --infer_num_epochs=1 --high_sampling_power=1 --label_noise=0.001 > logs/spucomnist_spare_power1_label_noise_0.001_1.log &
# python src/spucomnist_spare.py --seed=3 --gpu=5 --infer_num_epochs=1 --high_sampling_power=1 --label_noise=0.001 > logs/spucomnist_spare_power1_label_noise_0.001_2.log &

# python src/spucomnist_spare.py --seed=0 --gpu=7 --infer_num_epochs=1 --high_sampling_power=1 --label_noise=0.00001 > logs/spucomnist_spare_power1_label_noise_0.00001_0.log &
# python src/spucomnist_spare.py --seed=1 --gpu=6 --infer_num_epochs=1 --high_sampling_power=1 --label_noise=0.00001 > logs/spucomnist_spare_power1_label_noise_0.00001_1.log &
# python src/spucomnist_spare.py --seed=3 --gpu=5 --infer_num_epochs=1 --high_sampling_power=1 --label_noise=0.00001 > logs/spucomnist_spare_power1_label_noise_0.00001_2.log &

# python src/spucomnist_spare.py --seed=0 --gpu=4 --infer_num_epochs=1 --high_sampling_power=1 --feature_noise=0.1 > logs/spucomnist_spare_power1_feature_noise_0.1_0.log & 
# python src/spucomnist_spare.py --seed=1 --gpu=3 --infer_num_epochs=1 --high_sampling_power=1 --feature_noise=0.1 > logs/spucomnist_spare_power1_feature_noise_0.1_1.log &
# python src/spucomnist_spare.py --seed=3 --gpu=2 --infer_num_epochs=1 --high_sampling_power=1 --feature_noise=0.1 > logs/spucomnist_spare_power1_feature_noise_0.1_2.log &

# python src/spucomnist_spare.py --seed=0 --gpu=4 --infer_num_epochs=1 --high_sampling_power=1 --feature_noise=0.2 > logs/spucomnist_spare_power1_feature_noise_0.2_0.log &
# python src/spucomnist_spare.py --seed=1 --gpu=3 --infer_num_epochs=1 --high_sampling_power=1 --feature_noise=0.2 > logs/spucomnist_spare_power1_feature_noise_0.2_1.log &
# python src/spucomnist_spare.py --seed=3 --gpu=2 --infer_num_epochs=1 --high_sampling_power=1 --feature_noise=0.2 > logs/spucomnist_spare_power1_feature_noise_0.2_2.log &

# python src/spucomnist_spare.py --seed=0 --gpu=7 --lr=1e-3 --weight_decay=5e-3 --infer_num_epochs=10 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=magnitude_medium > logs/spucomnist_spare_power1_medium_9.log & 
# python src/spucomnist_spare.py --seed=1 --gpu=6 --lr=1e-3 --weight_decay=5e-3 --infer_num_epochs=10 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=magnitude_medium > logs/spucomnist_spare_power1_medium_10.log &
# python src/spucomnist_spare.py --seed=2 --gpu=5 --lr=1e-3 --weight_decay=5e-3 --infer_num_epochs=10 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=magnitude_medium > logs/spucomnist_spare_power1_medium_11.log &

# python src/spucomnist_spare.py --seed=0 --gpu=4 --lr=1e-3 --weight_decay=5e-4 --infer_lr=1e-2 --infer_weight_decay=5e-4 --infer_num_epochs=15 --high_sampling_power=1 --feature_difficulty=magnitude_small > logs/spucomnist_spare_power1_hard_wd5e-4_5_3.log & 
# python src/spucomnist_spare.py --seed=1 --gpu=3 --lr=1e-3 --weight_decay=5e-4 --infer_lr=1e-2 --infer_weight_decay=5e-4 --infer_num_epochs=15 --high_sampling_power=1 --feature_difficulty=magnitude_small > logs/spucomnist_spare_power1_hard_wd5e-4_5_4.log &
# python src/spucomnist_spare.py --seed=2 --gpu=2 --lr=1e-3 --weight_decay=5e-4 --infer_lr=1e-2 --infer_weight_decay=5e-4 --infer_num_epochs=15 --high_sampling_power=1 --feature_difficulty=magnitude_small > logs/spucomnist_spare_power1_hard_wd5e-4_5_5.log &

# python src/spucomnist_spare.py --seed=0 --gpu=4 --lr=1e-3 --weight_decay=5e-4 --infer_lr=1e-2 --infer_weight_decay=5e-4 --infer_num_epochs=20 --high_sampling_power=1 --feature_difficulty=magnitude_small > logs/spucomnist_spare_power1_hard_wd5e-4_10_3.log & 
# python src/spucomnist_spare.py --seed=1 --gpu=3 --lr=1e-3 --weight_decay=5e-4 --infer_lr=1e-2 --infer_weight_decay=5e-4 --infer_num_epochs=20 --high_sampling_power=1 --feature_difficulty=magnitude_small > logs/spucomnist_spare_power1_hard_wd5e-4_10_4.log &
# python src/spucomnist_spare.py --seed=2 --gpu=2 --lr=1e-3 --weight_decay=5e-4 --infer_lr=1e-2 --infer_weight_decay=5e-4 --infer_num_epochs=20 --high_sampling_power=1 --feature_difficulty=magnitude_small > logs/spucomnist_spare_power1_hard_wd5e-4_10_5.log &


# python src/spucomnist_spare.py --seed=0 --gpu=7 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=5 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_low > logs/spucomnist_spare_power1_var_low_3.log & 
# python src/spucomnist_spare.py --seed=1 --gpu=6 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=5 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_low > logs/spucomnist_spare_power1_var_low_4.log &
# python src/spucomnist_spare.py --seed=2 --gpu=5 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=5 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_low > logs/spucomnist_spare_power1_var_low_5.log &

# python src/spucomnist_spare.py --seed=0 --gpu=4 --lr=1e-3 --weight_decay=1e-4 --infer_num_epochs=5 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_medium > logs/spucomnist_spare_power1_var_med_0.log & 
# python src/spucomnist_spare.py --seed=1 --gpu=3 --lr=1e-3 --weight_decay=1e-4 --infer_num_epochs=5 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_medium > logs/spucomnist_spare_power1_var_med_1.log &
# python src/spucomnist_spare.py --seed=2 --gpu=2 --lr=1e-3 --weight_decay=1e-4 --infer_num_epochs=5 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_medium > logs/spucomnist_spare_power1_var_med_2.log &

python src/spucomnist_spare.py --seed=0 --gpu=4 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=15 --infer_lr=1e-2 --infer_weight_decay=5e-4 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_0.log & 
python src/spucomnist_spare.py --seed=1 --gpu=3 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=15 --infer_lr=1e-2 --infer_weight_decay=5e-4 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_1.log &
python src/spucomnist_spare.py --seed=2 --gpu=2 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=15 --infer_lr=1e-2 --infer_weight_decay=5e-4 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_2.log &

python src/spucomnist_spare.py --seed=0 --gpu=4 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=5e-4 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_4.log & 
python src/spucomnist_spare.py --seed=1 --gpu=3 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=5e-4 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_5.log &
python src/spucomnist_spare.py --seed=2 --gpu=2 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=5e-4 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_6.log &


# python src/spucomnist_spare.py --seed=0 --gpu=4 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=10 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_3.log & 
# python src/spucomnist_spare.py --seed=1 --gpu=3 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=10 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_4.log &
# python src/spucomnist_spare.py --seed=2 --gpu=2 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=10 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_5.log &

# python src/spucomnist_spare.py --seed=0 --gpu=4 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=9 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_9.log & 
# python src/spucomnist_spare.py --seed=1 --gpu=3 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=9 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_10.log &
# python src/spucomnist_spare.py --seed=2 --gpu=2 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=9 --infer_lr=1e-2 --high_sampling_power=1 --feature_difficulty=variance_high > logs/spucomnist_spare_power1_var_high_11.log &