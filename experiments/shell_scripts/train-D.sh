#! /bin/bash
experiment_name=D
n_systems=40
n_samples=5000
lr=0.0001
epochs=10000
print_freq=500

python ../train.py --hidden_dims 250 --experiment_name ${experiment_name} \
--n_systems ${n_systems} --n_samples ${n_samples} --lr ${lr} --epochs ${epochs} \
--print_freq ${print_freq}
