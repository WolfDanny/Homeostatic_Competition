cd ../Model-2/QSD-S-2/ 
#$ -V 
#$ -l node_type=40core-768G 
#$ -pe smp 5 
#$ -l h_rt=01:30:00 
#$ -l h_vmem=24G 
#$ -m ea 
#$ -M mmdfld@leeds.ac.uk 
source activate fs 
python Homeostatic_QSD.py
