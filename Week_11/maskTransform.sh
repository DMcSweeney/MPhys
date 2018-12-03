echo \# Bash Script that transforms planning masks into PET masks

cwd="/mnt/e/Mphys/"
mask_out_dir="/mnt/e/Mphys/PatientMasks"
mask_in_dir="/mnt/e/Mphys/PatientMasks_Backup"
base_dir="/mnt/e/Mphys/NiftyPatients"
fixed_ref="PlanningCT"
moving_ref="PET"
reg_out_dir="/mnt/e/Mphys/InverseElastixReg"

if [ -d ${mask_out_dir}  ]; then
  rm -rf ${mask_out_dir}
fi
if [ -d ${reg_out_dir}  ]; then
  rm -rf ${reg_out_dir}
fi
#  Create directories for Inverse Registration Images
mkdir ${reg_out_dir}
mkdir ${reg_out_dir}/Rigid/
mkdir ${reg_out_dir}/Non-Rigid/
mkdir ${reg_out_dir}/DVF/
# Create directories for Mask Output
mkdir ${mask_out_dir}
mkdir ${mask_out_dir}/PET/
mkdir ${mask_out_dir}/PET/Rigid/
mkdir ${mask_out_dir}/PET/Non-Rigid/

for filepath in `(ls -f ${mask_in_dir}/${fixed_ref}/*.nii )`; do
  #  Get patient name from filename
  filename=$(basename -- "$filepath")
  image="${filename%.*}"
  echo $image
  mkdir ${reg_out_dir}/Rigid/${image}
  mkdir ${reg_out_dir}/Non-Rigid/${image}
  mkdir ${reg_out_dir}/DVF/${image}
  mkdir ${mask_out_dir}/PET/Rigid/${image}
  mkdir ${mask_out_dir}/PET/Non-Rigid/${image}
  # Perform Registration, first affine then b-spline
  elastix -m ${base_dir}/${fixed_ref}/${image} -f ${base_dir}/${moving_ref}/${image} -p ${cwd}/Affine-parameter-file.txt  -out ${reg_out_dir}/Rigid/${image}
  transformix -in ${mask_in_dir}/${fixed_ref}/${image} -out ${mask_out_dir}/${moving_ref}/Rigid/${image} -tp ${reg_out_dir}/Rigid/${image}/TransformParameters.0.txt
  elastix -m ${reg_out_dir}/Rigid/${image}/result.0.nii -f ${base_dir}/${moving_ref}/${image} -p ${cwd}/B-Spline-parameter-file.txt  -out ${reg_out_dir}/Non-Rigid/${image} 
  # Generate deformation field from affine transform file
  transformix -def all -tp ${reg_out_dir}/Non-Rigid/${image}/TransformParameters.0.txt -out ${reg_out_dir}/DVF/${image}
  transformix -in ${mask_out_dir}/${moving_ref}/Rigid/${image}/result.nii -out ${mask_out_dir}/${moving_ref}/Non-Rigid/${image} -tp ${reg_out_dir}/Non-Rigid/${image}/TransformParameters.0.txt
done
