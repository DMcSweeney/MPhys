#!/bin/sh


if [ ! -d NiftyPatients/  ]; then
  mkdir /mnd/e/Mphys/NiftyPatients/
  mkdir /mnd/e/Mphys/NiftyPatients/PET/
  mkdir /mnd/e/Mphys/NiftyPatients/PlanningCT/
fi

python3 ./Week_7/writeNifty.py --patient_dir '/mnd/e/Mphys/Patients/' --pet_outdir '/mnd/e/Mphys/NiftyPatients/PET/' --planning_outdir '/mnd/e/Mphys/NiftyPatients/PlanningCT/'
sh ./Week_7/elastix.sh
