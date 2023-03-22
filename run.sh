#!/bin/bash
#SBATCH --job-name=mighty            # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --partition=cpu_normal # Partition auf der gerechnet werden soll (Bei GPU Jobs unbedingt notwendig)
#SBATCH --mem=1G                       # Reservierung von 1 GB RAM Speicher pro Knoten

source ~/anaconda3/tmp/bin/activate mighty

python mighty/run_mighty.py --out-dir ./fd --episodes 10000 --seed $1
