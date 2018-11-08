#!/bin/sh

cwd="/mnt/e/Mphys/"
echo cwd

cd /mnt/e/Mphys/

if [ ! -d NiftyPatients/  ]; then
  mkdir NiftyPatients/
  mkdir NiftyPatients/PET/
  mkdir NiftyPatients/PlanningCT/
fi

python3 /Week_7/writeNifty.py --patient_dir 'Patients/' --pet_outdir 'NiftyPatients/PET/' --planning_outdir 'NiftyPatients/PlanningCT/'

sh /Week_7/elastix.sh
