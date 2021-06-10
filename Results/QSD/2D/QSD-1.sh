#$ -V 
#$ -pe smp 5 
#$ -l h_rt=01:30:00 
#$ -l h_vmem=2G 
#$ -m ea 
#$ -M mmdfld@leeds.ac.uk 
source activate fs 
python Homeostatic_QSD_1.py

