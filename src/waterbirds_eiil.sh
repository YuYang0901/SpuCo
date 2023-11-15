# python src/waterbirds_eiil.py --seed=0 --gpu=5 --pretrained > logs/waterbirds_eiil_reinit_v1_2groups_0.log &

# sleep 10

# python src/waterbirds_eiil.py --seed=1 --gpu=6 --pretrained > logs/waterbirds_eiil_reinit_v1_2groups_1.log &

# sleep 10

# python src/waterbirds_eiil.py --seed=2 --gpu=7 --pretrained > logs/waterbirds_eiil_reinit_v1_2groups_2.log &

python src/waterbirds_eiil.py --lr 1e-4 --weight_decay 1e-4 --seed=0 --gpu=5 > logs/waterbirds_eiil_from_scratch_3.log &

sleep 10

python src/waterbirds_eiil.py --lr 1e-4 --weight_decay 1e-4 --seed=1 --gpu=6 --infer_num_epochs=30 > logs/waterbirds_eiil_from_scratch_4.log &

sleep 10

python src/waterbirds_eiil.py --lr 1e-4 --weight_decay 1e-4 --seed=2 --gpu=7 > logs/waterbirds_eiil_from_scratch_5.log &