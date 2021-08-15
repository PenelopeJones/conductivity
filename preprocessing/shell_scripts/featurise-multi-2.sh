#! /bin/bash
conc=0.035
print_freq=2

for lb in 2.5 5.0 8.0; do
  python ../featurise_static.py --conc ${conc} --lb ${lb} \
  --print_freq ${print_freq}
done
