#!/bin/sh
echo \# Bash script to perform registration using elastix

cwd="/mnt/d/Documents/GitHub/MPhys"
base_dir="/mnt/d/Documents/GitHub/MPhys/NiftyPatients"
fixed_ref="PlanningCT"
moving_ref="PET"
out_dir="/mnt/e/Mphys/ElastixReg"

for filepath in `(ls -f ${base_dir}/${fixed_ref}/*.nii )`; do

  filename=$(basename -- "$filepath")
  image="${filename%.*}"
  echo $image
  mkdir ${out_dir}/Rigid/
  mkdir ${out_dir}/Non-Rigid/
  mkdir ${out_dir}/Rigid/${image}
  mkdir ${out_dir}/Non-Rigid/${image}

  elastix -f ${base_dir}/${fixed_ref}/${image} -m ${base_dir}/${moving_ref}/${image} -p ${cwd}/Affine-parameter-file.txt  -out ${out_dir}/Rigid/${image}
  elastix -f ${base_dir}/${fixed_ref}/${image} -m ${out_dir}/Rigid/${image}/result.0.nii -p ${cwd}/B-Spline-parameter-file.txt  -out ${out_dir}/Non-Rigid/${image}
done
