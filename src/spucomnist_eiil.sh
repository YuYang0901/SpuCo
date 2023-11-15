python src/spucomnist_eiil.py --seed=0 --gpu=4 --lr=1e-2 --infer_num_epochs=20 --infer_lr=1e-3 --feature_difficulty=magnitude_large > logs/spucomnist_eiil_easy_0.log & 
python src/spucomnist_eiil.py --seed=1 --gpu=4 --lr=1e-2 --infer_num_epochs=20 --infer_lr=1e-3 --feature_difficulty=magnitude_large > logs/spucomnist_eiil_easy_1.log &
python src/spucomnist_eiil.py --seed=2 --gpu=4 --lr=1e-2 --infer_num_epochs=20 --infer_lr=1e-3 --feature_difficulty=magnitude_large > logs/spucomnist_eiil_easy_2.log &

# python src/spucomnist_eiil.py --seed=0 --gpu=0 --lr=1e-3 --infer_num_epochs=20 --infer_lr=1e-3 --feature_difficulty=magnitude_medium > logs/spucomnist_eiil_medium_0.log & 
# python src/spucomnist_eiil.py --seed=1 --gpu=1 --lr=1e-3 --infer_num_epochs=20 --infer_lr=1e-3 --feature_difficulty=magnitude_medium > logs/spucomnist_eiil_medium_1.log &
# python src/spucomnist_eiil.py --seed=2 --gpu=2 --lr=1e-3 --infer_num_epochs=20 --infer_lr=1e-3 --feature_difficulty=magnitude_medium > logs/spucomnist_eiil_medium_2.log &

# python src/spucomnist_eiil.py --seed=0 --gpu=0 --lr=1e-3 --infer_lr=1e-3 --infer_num_epochs=20 --feature_difficulty=magnitude_small > logs/spucomnist_eiil_hard_0.log & 
# python src/spucomnist_eiil.py --seed=1 --gpu=1 --lr=1e-3 --infer_lr=1e-3 --infer_num_epochs=20 --feature_difficulty=magnitude_small > logs/spucomnist_eiil_hard_1.log &
# python src/spucomnist_eiil.py --seed=2 --gpu=2 --lr=1e-3 --infer_lr=1e-3 --infer_num_epochs=20 --feature_difficulty=magnitude_small > logs/spucomnist_eiil_hard_2.log &


# python src/spucomnist_eiil.py --seed=0 --gpu=4 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=1e-2 --feature_difficulty=variance_low > logs/spucomnist_eiil_var_low_0.log & 
# python src/spucomnist_eiil.py --seed=1 --gpu=4 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=1e-2 --feature_difficulty=variance_low > logs/spucomnist_eiil_var_low_1.log &
# python src/spucomnist_eiil.py --seed=2 --gpu=5 --lr=1e-2 --weight_decay=5e-3 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=1e-2 --feature_difficulty=variance_low > logs/spucomnist_eiil_var_low_2.log &

# python src/spucomnist_eiil.py --seed=0 --gpu=3 --lr=1e-3 --weight_decay=1e-3 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=1e-2 --feature_difficulty=variance_medium > logs/spucomnist_eiil_var_med_0.log & 
# python src/spucomnist_eiil.py --seed=1 --gpu=4 --lr=1e-3 --weight_decay=1e-3 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=1e-2 --feature_difficulty=variance_medium > logs/spucomnist_eiil_var_med_1.log &
# python src/spucomnist_eiil.py --seed=2 --gpu=5 --lr=1e-3 --weight_decay=1e-3 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=1e-2 --feature_difficulty=variance_medium > logs/spucomnist_eiil_var_med_2.log &

# python src/spucomnist_eiil.py --seed=0 --gpu=3 --lr=1e-2 --weight_decay=1e-2 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=1e-2 --feature_difficulty=variance_high > logs/spucomnist_eiil_var_high_0.log & 
# python src/spucomnist_eiil.py --seed=1 --gpu=4 --lr=1e-2 --weight_decay=1e-2 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=1e-2 --feature_difficulty=variance_high > logs/spucomnist_eiil_var_high_1.log &
# python src/spucomnist_eiil.py --seed=3 --gpu=5 --lr=1e-2 --weight_decay=1e-2 --infer_num_epochs=20 --infer_lr=1e-2 --infer_weight_decay=1e-2 --feature_difficulty=variance_high > logs/spucomnist_eiil_var_high_2.log &
