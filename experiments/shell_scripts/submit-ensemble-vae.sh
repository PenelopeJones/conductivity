#! /bin/bash
experiment_name=ENSEMBLE_VAE
lr=0.01
epochs=100
print_freq=10
n_splits=2
n_ensembles=2
ae_epochs=100

mkdir ../results/$experiment_name
mkdir ../results/${experiment_name}/models
mkdir ../results/${experiment_name}/predictions

for n_split in {0..1}
do
  for run in {0..1}
  do
    sbatch --time=1:00:00 -J soap-ensemble --export=script="../train-vae.py",kwargs="--experiment_name $experiment_name --lr $lr --ae_epochs $ae_epochs --epochs $epochs --print_freq $print_freq --n_split $n_split --run_id $run --n_splits $n_splits --n_ensembles $n_ensembles" run-soap-gpu
  done
done
