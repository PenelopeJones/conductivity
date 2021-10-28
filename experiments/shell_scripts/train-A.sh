#! /bin/bash
experiment_name=MPI_A
n_systems=5
n_samples=500
lr=0.001
epochs=500
print_freq=20

mpirun -n 4 ../train-soap-mpi.py --hidden_dims 100 --experiment_name ${experiment_name} \
--n_systems ${n_systems} --n_samples ${n_samples} --lr ${lr} --epochs ${epochs} \
--print_freq ${print_freq}
