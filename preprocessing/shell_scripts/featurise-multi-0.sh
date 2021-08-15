#! /bin/bash
conc=0.035
print_freq=2

for lb in 4.0 7.0 10.0; do
  python ../featurise_static.py --conc ${conc} --lb ${lb} \
  --print_freq ${print_freq}
done
