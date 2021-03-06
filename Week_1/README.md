# Week beginning 24/09
Made Git Repo for project and introduction to machine learning and Keras. <br>


PetCT, PCT and Vector field info is stored in DICOM format so need to read this into Python. 
Use [pyDICOM](https://github.com/pydicom/pydicom) or [simpleITK](http://www.simpleitk.org/).

SimpleITK is a C++ library with binary distribution for Python whereas pyDICOM is a python library.<br>
SimpleITK could be faster since backend is in C but it looks more convoluted to implement. See [here](https://github.com/concept-to-clinic/concept-to-clinic/issues/121 "SimpleITK vs pyDICOM").<br>

All data is open-source and can be found at: https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT <br>
I think this is the dataset they showed us.<br>

The code for Voxel Morph is available [here](https://github.com/voxelmorph/voxelmorph "Voxelmorph").<br>

*__Links__*<br>
[Voxel Morph](https://arxiv.org/pdf/1809.05231.pdf) - This paper expands on the authors' previous work [here](https://arxiv.org/pdf/1802.02604.pdf)<br>
[FCN](https://arxiv.org/ftp/arxiv/papers/1709/1709.00799.pdf) - Ref. 37 in Voxel Morph paper<br>
[CNN IR](https://wbir2018.nl/files/WBIR2018_Abstracts.pdf)<br>
[CNN-IR-2](https://pure.tue.nl/ws/portalfiles/portal/98728122/105740S.pdf)<br>
[IR in Radiotherapy](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.12256)<br>
[Non-rigid IR](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5518453/)<br>






