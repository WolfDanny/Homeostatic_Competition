# QSD Established clones [1=2=3]

- Matrix-established-0.csv

t=1:00:00
m=5G
c=5
name_python=FS-absorption-probability
name_folder=FS-AP
name_shell=FS-AP-est

# Creating and populating a directory for each state
'''bash
for ((j=0;j<=2;j++));
do mkdir -p $name_folder-$j;
cp $name_python.py $name_folder-$j/;
Samp="$j";
sed -i 's/SampleHolder/'$Samp'/g' $name_folder-$j/$name_python.py;
echo -e "cd ../$name_folder-$j/ \n#$ -V \n#$ -pe smp $c \n#$ -l h_rt=$t \n#$ -l h_vmem=$m \n#$ -m ea \nsource activate fs \npython $name_python.py" > "$name_folder-$j/$name_shell-$j.sh";
done
'''

# Submitting all jobs from Outfiles directory
'''bash
for ((j=0;j<=2;j++));
do qsub ../$name_folder-$j/$name_shell-$j.sh;
done
'''

# QSD Close extinction balanced [1<2<3]

- Matrix-extinction-balanced-0.csv

t=1:00:00
m=5G
c=5
name_python=FS-absorption-probability
name_folder=FS-AP
name_shell=FS-AP-ext-bal

# Creating and populating a directory for each state
'''bash
for ((j=0;j<=2;j++));
do mkdir -p $name_folder-$j;
cp $name_python.py $name_folder-$j/;
Samp="$j";
sed -i 's/SampleHolder/'$Samp'/g' $name_folder-$j/$name_python.py;
echo -e "cd ../$name_folder-$j/ \n#$ -V \n#$ -pe smp $c \n#$ -l h_rt=$t \n#$ -l h_vmem=$m \n#$ -m ea \nsource activate fs \npython $name_python.py" > "$name_folder-$j/$name_shell-$j.sh";
done
'''

# Submitting all jobs from Outfiles directory
'''bash
for ((j=0;j<=2;j++));
do qsub ../$name_folder-$j/$name_shell-$j.sh;
done
'''

# QSD Close extinction competition [1=2<3]

- Matrix-extinction-competition-0.csv

t=1:00:00
m=5G
c=5
name_python=FS-absorption-probability
name_folder=FS-AP
name_shell=FS-AP-ext-comp

# Creating and populating a directory for each state
'''bash
for ((j=0;j<=3;j++));
do mkdir -p $name_folder-$j;
cp $name_python.py $name_folder-$j/;
Samp="$j";
sed -i 's/SampleHolder/'$Samp'/g' $name_folder-$j/$name_python.py;
echo -e "cd ../$name_folder-$j/ \n#$ -V \n#$ -pe smp $c \n#$ -l h_rt=$t \n#$ -l h_vmem=$m \n#$ -m ea \nsource activate fs \npython $name_python.py" > "$name_folder-$j/$name_shell-$j.sh";
done
'''

# Submitting all jobs from Outfiles directory
'''bash
for ((j=0;j<=3;j++));
do qsub ../$name_folder-$j/$name_shell-$j.sh;
done
'''