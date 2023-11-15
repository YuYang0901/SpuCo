python src/spucoanimals_eiil.py --seed=0 --gpu=6 --lr=1e-4 --weight_decay=1e-4 --infer_num_epochs=-1 > logs/spucoanimals_eiil_0.log &

sleep 10

python src/spucoanimals_eiil.py --seed=1 --gpu=6 --lr=1e-4 --weight_decay=1e-4 --infer_num_epochs=-1 > logs/spucoanimals_eiil_1.log &

sleep 10

python src/spucoanimals_eiil.py --seed=2 --gpu=6 --lr=1e-4 --weight_decay=1e-4 --infer_num_epochs=-1 > logs/spucoanimals_eiil_2.log &