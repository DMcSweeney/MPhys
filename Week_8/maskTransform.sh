echo \# Bash Script that transforms planning masks into PET masks

mask_out_dir="/mnt/e/Mphys/PatientMasks"
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

mkdir ${reg_out_dir}
mkdir ${reg_out_dir}/PET/
mkdir ${mask_out_dir}/PET/

for filepath in `(ls -f ${out_dir}/${fixed_ref}/*.nii )`; do
  #  Get patient name from filename
  filename=$(basename -- "$filepath")
  image="${filename%.*}"
  echo $image
  mkdir ${out_dir}/PET/${image}
  # Perform Registration, first affine then b-spline
  elastix -m ${base_dir}/${fixed_ref}/${image} -f ${base_dir}/${moving_ref}/${image} -p ${cwd}/Affine-parameter-file.txt  -out ${reg_out_dir}/Rigid/${image}
  elastix -m ${base_dir}/${fixed_ref}/${image} -f ${out_dir}/Rigid/${image}/result.0.nii -p ${cwd}/B-Spline-parameter-file.txt  -out ${out_dir}/Non-Rigid/${image}
  # Generate deformation field from affine transform file
  transformix -def all -tp ${out_dir}/Non-Rigid/${image}/TransformParameters.0.txt -out ${out_dir}/DVF/${image}
done
