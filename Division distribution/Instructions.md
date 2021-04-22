# QSD Established clones [1=2=3]

- Matrix-established-0.csv

t=0:20:00
m=5G
c=5
name_python=FS-division-distribution
name_folder=FS-DD
name_shell=FS-DD-est

# Creating and populating a directory for each state
'''bash
for ((i=0;i<=2;i++));
do mkdir -p Clone-$i;
for ((j=0;j<=2;j++));
do mkdir -p Clone-$i/$name_folder-$j;
cp $name_python.py Clone-$i/$name_folder-$j/;
Clon="$i";
Samp="$j";
sed -i 's/SampleHolder/'$Samp'/g' Clone-$i/$name_folder-$j/$name_python.py;
sed -i 's/CloneHolder/'$Clon'/g' Clone-$i/$name_folder-$j/$name_python.py;
echo -e "cd ../Clone-$i/$name_folder-$j/ \n#$ -V \n#$ -pe smp $c \n#$ -l h_rt=$t \n#$ -l h_vmem=$m \n#$ -m ea \nsource activate fs \npython $name_python.py" > "Clone-$i/$name_folder-$j/$name_shell-$j.sh";
done;
done
'''

# Submitting all jobs from Outfiles directory
'''bash
for ((i=0;i<=2;i++));
do for ((j=0;j<=2;j++));
do qsub ../Clone-$i/$name_folder-$j/$name_shell-$j.sh;
done;
done
'''

# QSD Close extinction balanced [1<2<3]

- Matrix-extinction-balanced-0.csv

t=1:30:00
m=5G
c=5
name_python=FS-division-distribution
name_folder=FS-DD
name_shell=FS-DD-ext-bal

# Creating and populating a directory for each state
'''bash
for ((i=0;i<=2;i++));
do mkdir -p Clone-$i;
for ((j=0;j<=2;j++));
do mkdir -p Clone-$i/$name_folder-$j;
cp $name_python.py Clone-$i/$name_folder-$j/;
Clon="$i";
Samp="$j";
sed -i 's/SampleHolder/'$Samp'/g' Clone-$i/$name_folder-$j/$name_python.py;
sed -i 's/CloneHolder/'$Clon'/g' Clone-$i/$name_folder-$j/$name_python.py;
echo -e "cd ../Clone-$i/$name_folder-$j/ \n#$ -V \n#$ -pe smp $c \n#$ -l h_rt=$t \n#$ -l h_vmem=$m \n#$ -m ea \nsource activate fs \npython $name_python.py" > "Clone-$i/$name_folder-$j/$name_shell-$j.sh";
done;
done
'''

# Submitting all jobs from Outfiles directory
'''bash
for ((i=0;i<=2;i++));
do for ((j=0;j<=2;j++));
do qsub ../Clone-$i/$name_folder-$j/$name_shell-$j.sh;
done;
done
'''

# QSD Close extinction competition [1=2<3]

- Matrix-extinction-competition-0.csv

t=1:30:00
m=5G
c=5
name_python=FS-division-distribution
name_folder=FS-DD
name_shell=FS-DD-ext-comp

# Creating and populating a directory for each state
'''bash
for ((i=0;i<=2;i++));
do mkdir -p Clone-$i;
for ((j=0;j<=3;j++));
do mkdir -p Clone-$i/$name_folder-$j;
cp $name_python.py Clone-$i/$name_folder-$j/;
Clon="$i";
Samp="$j";
sed -i 's/SampleHolder/'$Samp'/g' Clone-$i/$name_folder-$j/$name_python.py;
sed -i 's/CloneHolder/'$Clon'/g' Clone-$i/$name_folder-$j/$name_python.py;
echo -e "cd ../Clone-$i/$name_folder-$j/ \n#$ -V \n#$ -pe smp $c \n#$ -l h_rt=$t \n#$ -l h_vmem=$m \n#$ -m ea \nsource activate fs \npython $name_python.py" > "Clone-$i/$name_folder-$j/$name_shell-$j.sh";
done;
done
'''

# Submitting all jobs from Outfiles directory
'''bash
for ((i=0;i<=2;i++));
do for ((j=0;j<=3;j++));
do qsub ../Clone-$i/$name_folder-$j/$name_shell-$j.sh;
done;
done
'''