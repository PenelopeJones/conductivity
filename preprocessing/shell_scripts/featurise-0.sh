#! /bin/bash
conc=0.045
lb=10
print_freq=2

python ../featurise_static.py --conc ${conc} --lb ${lb} \
--print_freq ${print_freq}
