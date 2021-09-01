#! /bin/bash
experiment_name=B
n_systems=40
n_samples=10000
lr=0.00001
epochs=20000
print_freq=1000

python ../train.py --hidden_dims 50 50 --experiment_name ${experiment_name} \
--n_systems ${n_systems} --n_samples ${n_samples} --lr ${lr} --epochs ${epochs} \
--print_freq ${print_freq}
