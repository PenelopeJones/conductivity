#! /bin/bash
conc=0-0005
lb=2-5

for frac in 1 2 3 4
do
  mv label_dumbbells_0-${}_${conc}_${lb}_soap.npy labels/
done
