#! /bin/bash
print_freq=2

for conc in 0.0005 0.001 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05
do
  for lb in 2.5 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
  do
    sbatch --time=1:30:00 -J bp-$conc-$lb --export=script="../featurise_bp.py",kwargs="--conc $conc --lb $lb --print_freq $print_freq" run-bp-gpu
  done
done
