#!/bin/sh
echo \# Bash script to perform resampling to coarser image

# cwd="/mnt/e/Mphys/" # Dir from which script is run
base_dir="/hepgpu3-data1/dmcsween/data" # Directory containing files to resample
fixed_ref="PlanningCT"
moving_ref="PET_Rigid"
out_dir="/hepgpu3-data1/dmcsween/resample_data" #Dir to which we should write

if [ -d ${out_dir}  ]; then
  rm -rf ${out_dir}
fi
mkdir ${out_dir}
mkdir ${out_dir}/${moving_ref}
mkdir ${out_dir}/${fixed_ref}
mkdir ${out_dir}/Rigid/
mkdir ${out_dir}/Non-Rigid/
mkdir ${out_dir}/DVF/
# Loop over all patients saved in Nifty Patients
for filepath in `(ls -f ${base_dir}/${fixed_ref}/*.nii )`; do
  #  Get patient name from filename
  filename=$(basename -- "$filepath")
  image="${filename%.*}"
  echo $image
  # # Make required directories
  # mkdir ${out_dir}/DVF/${image}
  # mkdir ${out_dir}/Rigid/${image}
  # mkdir ${out_dir}/Non-Rigid/${image}
  #Resample Images
  python resample_image.py --input_image ${base_dir}/${fixed_ref}/${image} --output_image ${out_dir}/${fixed_ref}/${image}.nii
  python resample_image.py --input_image ${base_dir}/${moving_ref}/${image} --output_image ${out_dir}/${moving_ref}/${image}.nii

  # Perform Registration, first affine then b-spline
  # elastix -f ${base_dir}/${fixed_ref}/${image} -m ${base_dir}/${moving_ref}/${image} -p ${cwd}/Affine-parameter-file.txt  -out ${out_dir}/Rigid/${image}
  # elastix -f ${base_dir}/${fixed_ref}/${image} -m ${out_dir}/Rigid/${image}/result.0.nii -p ${cwd}/B-Spline-parameter-file.txt  -out ${out_dir}/Non-Rigid/${image}
  # # Generate deformation field from affine transform file
  # transformix -def all -tp ${out_dir}/Non-Rigid/${image}/TransformParameters.0.txt -out ${out_dir}/DVF/${image}
done
