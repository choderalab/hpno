#BSUB -q cpuqueue
#BSUB -o %J.stdout
#BSUB -R "rusage[mem=10] span[ptile=1]"
#BSUB -W 1:00
#BSUB -n 1

python generate_data.py


