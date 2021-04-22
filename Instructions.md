# General Instrucitons

1. Make sure that the stimulus vector has the correct values (check the brackets on the title of each section).
2. Change the file path to '../../Samples/filename'.
3. Make sure that the matrix file names match those of the case (check the name at the beginning of each section).

# QSD Established clones [1=2=3]

- Matrix-{}.csv

t=01:30:00
m=24G
c=5
name_python=Homeostatic_QSD
name_shell=QSD-S

## Creating and populating a directory for each state
'''bash
for ((i=1;i<=2;i++));
do mkdir -p Model-$i;
for ((j=0;j<=3;j++));
do mkdir -p Model-$i/$name_shell-$j;
cp $name_python.py Model-$i/$name_shell-$j/;
sed -i "s/SampleHolder/$j/g" Model-$i/$name_shell-$j/$name_python.py;
sed -i "s/ModelHolder/$i/g" Model-$i/$name_shell-$j/$name_python.py;
echo -e "cd ../Model-$i/$name_shell-$j/ \n#$ -V \n#$ -l node_type=40core-768G \n#$ -pe smp $c \n#$ -l h_rt=$t \n#$ -l h_vmem=$m \n#$ -m ea \n#$ -M mmdfld@leeds.ac.uk \nsource activate fs \npython $name_python.py" > "Model-$i/$name_shell-$j/$name_shell-$i-$j.sh";
done;
done
'''

## Submitting all jobs
'''bash
for ((i=1;i<=2;i++));
do for ((j=0;j<=3;j++));
do qsub ../Model-$i/$name_shell-$j/$name_shell-$i-$j.sh;
done;
done
'''

# QSD Close extinction balanced [1<2<3]

- Matrix-extinction-balanced-0.csv

t=01:30:00
m=24G
c=5
name_python=Homeostatic_QSD
name_shell=QSD-ext-bal

## Creating and populating a directory for each state
'''bash
for ((i=0;i<=1;i++));
do mkdir -p Model-$i;
for ((j=0;j<=2;j++));
do mkdir -p Model-$i/$name_shell-$j;
cp $name_python.py Model-$i/$name_shell-$j/;
Samp="$j";
Mod="$i";
sed -i 's/SampleHolder/'$Samp'/g' Model-$i/$name_shell-$j/$name_python.py;
sed -i 's/ModelHolder/'$Mod'/g' Model-$i/$name_shell-$j/$name_python.py;
echo -e "cd ../Model-$i/$name_shell-$j/ \n#$ -V \n#$ -l node_type=40core-768G \n#$ -pe smp $c \n#$ -l h_rt=$t \n#$ -l h_vmem=$m \n#$ -m ea \n#$ -M mmdfld@leeds.ac.uk \nsource activate fs \npython $name_python.py" > "Model-$i/$name_shell-$j/$name_shell-$i-$j.sh";
done;
done
'''

## Submitting all jobs
'''bash
for ((i=0;i<=1;i++));
do for ((j=0;j<=2;j++));
do qsub ../Model-$i/$name_shell-$j/$name_shell-$i-$j.sh;
done;
done
'''

# QSD Close extinction competition [1=2<3]

- Matrix-extinction-competition-0.csv

t=01:30:00
m=24G
c=5
name_python=Homeostatic_QSD
name_shell=QSD-ext-comp

## Creating and populating a directory for each state
'''bash
for ((i=0;i<=1;i++));
do mkdir -p Model-$i;
for ((j=0;j<=3;j++));
do mkdir -p Model-$i/$name_shell-$j;
cp $name_python.py Model-$i/$name_shell-$j/;
Samp="$j";
Mod="$i";
sed -i 's/SampleHolder/'$Samp'/g' Model-$i/$name_shell-$j/$name_python.py;
sed -i 's/ModelHolder/'$Mod'/g' Model-$i/$name_shell-$j/$name_python.py;
echo -e "cd ../Model-$i/$name_shell-$j/ \n#$ -V \n#$ -l node_type=40core-768G \n#$ -pe smp $c \n#$ -l h_rt=$t \n#$ -l h_vmem=$m \n#$ -m ea \n#$ -M mmdfld@leeds.ac.uk \nsource activate fs \npython $name_python.py" > "Model-$i/$name_shell-$j/$name_shell-$i-$j.sh";
done;
done
'''

## Submitting all jobs
'''bash
for ((i=0;i<=1;i++));
do for ((j=0;j<=3;j++));
do qsub ../Model-$i/$name_shell-$j/$name_shell-$i-$j.sh;
done;
done
'''