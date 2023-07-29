#!/bin/bash 


#SBATCH --account=didan
#SBATCH --job-name=optimal
#SBATCH -e optimal.e.%A
#SBATCH -o optimal.o.%A
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=96gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=evanmc@arizona.edu

# other common options. 
### means they are commented out
###SBATCH --account=PI
###SBATCH --partition=windfall
###SBATCH --gres=gpu:1
###SBATCH --mem-per-cpu=5gb
###SBATCH —-mem=1600gb
###SBATCH --mem=1600gb
###SBATCH --mail-type=ALL
###SBATCH --mail-user=netid@email.arizona.edu

module load anaconda/2020
ulimit -S -c unlimited
source ~/.bash_profile
#conda init bash
conda activate weeds
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error in activation"
    exit $retVal
fi
#conda info 
#echo $PATH
#which python
cd ../jetson
python Selection.py -df ./normalized.csv -fs ALL -f 10 -o -l 
