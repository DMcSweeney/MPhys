import os
import numpy as np
import SimpleITK as sitk
import pydicom
from dicom_contour.contour import *

# Change working directory such that it can access data
os.chdir("..")
# Print current working directory
cwd = os.getcwd()
print(cwd)
contour_path = 'D:/Documents/GitHub/Patients/HN-CHUM-001/08-27-1885-TomoTherapy Patient Disease-00441/114120634-TomoTherapy Structure Set-68567/'
image_path = './Patients/HN-CHUM-001/08-27-1885-TomoTherapy Patient Disease-00441/112161818-kVCT Image Set-62659/'
patient = image_path.split('/')[2]
print(patient)


class ReadMasks(object):

    def get_contour_file(self, path):
        """
        Get contour file from a given path by searching for ROIContourSequence
        inside dicom data structure.
        More information on ROIContourSequence available here:
        http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html

        Inputs:
                path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient. Must contain contour file.
        Return:
            contour_file (str): name of the file with the contour
        """
        contour_file = ''
        # handle `/` missing
        if path[-1] != '/':
            path += '/'
        # get .dcm contour file
        fpaths = [path + f for f in os.listdir(path) if '.dcm' in f]
        print(fpaths)
        n = 0
        for fpath in fpaths:
            f = pydicom.read_file(fpath)
            print(type(f))
            if 'ROIContourSequence' in dir(f):
                contour_file = fpath.split('/')[-1]
                n += 1
            if n > 1:
                warnings.warn("There are multiple files, returning the last one!")
        print(contour_file)
        print("Got Contour File")
        return contour_file

    def get_data(self, contour_path, image_path, index):
        """
        Generate image array and contour array
        Inputs:
            path (str): path of the the directory that has DICOM files in it
            contour_dict (dict): dictionary created by get_contour_dict
            index (int): index of the desired ROISequence
        Returns:
            images and contours np.arrays
        """
        images = []
        contours = []
        # handle `/` missing
        if contour_path[-1] != '/':
            contour_path += '/'
        if image_path[-1] != '/':
            image_path += '/'
        # get contour file
        contour_file = self.get_contour_file(contour_path)
        # Find contour data
        contour_data = pydicom.read_file(contour_path + '/' + contour_file)
        # Get ROI names
        print(get_roi_names(contour_data))
        # get slice orders
        ordered_slices = self.slice_order(image_path)
        print(ordered_slices[:5])
        # get contour dict
        contour_dict = self.get_contour_dict(contour_file, contour_path, image_path, index)

        for k, v in ordered_slices:
            # get data from contour dict
            if k in contour_dict:
                images.append(contour_dict[k][0])
                contours.append(contour_dict[k][1])
            # get data from dicom.read_file
            else:
                img_arr = pydicom.read_file(path + k + '.dcm').pixel_array
                contour_arr = np.zeros_like(img_arr)
                images.append(img_arr)
                contours.append(contour_arr)

        return np.array(images), np.array(contours)

    def slice_order(self, path):
        """
        Takes path of directory that has the DICOM images and returns
        a ordered list that has ordered filenames
        Inputs
            path: path that has .dcm images
        Returns
            ordered_slices: ordered tuples of filename and z-position
        """
        # handle `/` missing
        if path[-1] != '/':
            path += '/'
        slices = []
        for s in os.listdir(path):
            try:
                f = pydicom.read_file(path + '/' + s)
                f.pixel_array  # to ensure not to read contour file
                slices.append(f)
            except:
                continue

        slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
        ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
        return ordered_slices

    def get_contour_dict(self, contour_file, contour_path, image_path, index):
        """
        Returns a dictionary as k: img fname, v: [corresponding img_arr, corresponding contour_arr]
        Inputs:
            contour_file: .dcm contour file name
            path: path which has contour and image files
        Returns:
            contour_dict: dictionary with 2d np.arrays
        """
        # handle `/` missing
        if contour_path[-1] != '/':
            contour_path += '/'
        if image_path[-1] != '/':
            image_path += '/'
        # img_arr, contour_arr, img_fname
        contour_list = self.cfile2pixels(contour_file, contour_path, image_path, index)

        contour_dict = {}
        for img_arr, contour_arr, img_id in contour_list:
            contour_dict[img_id] = [img_arr, contour_arr]

        return contour_dict

    def cfile2pixels(self, file, contour_path, image_path, ROIContourSeq=0):
        """
        Given a contour file and path of related images return pixel arrays for contours
        and their corresponding images.
        Inputs
            file: filename of contour
            path: path that has contour and image files
            ROIContourSeq: tells which sequence of contouring to use default 0 (RTV)
        Return
            contour_iamge_arrays: A list which have pairs of img_arr and contour_arr for a given contour file
        """
        # handle `/` missing
        if contour_path[-1] != '/':
            contour_path += '/'
        if image_path[-1] != '/':
            image_path += '/'
        f = pydicom.read_file(contour_path + file)
        # index 0 means that we are getting RTV information
        RTV = f.ROIContourSequence[ROIContourSeq]
        # get contour datasets in a list
        contours = [contour for contour in RTV.ContourSequence]
        print(len(contours))
        img_contour_arrays = [self.coord2pixels(cdata, image_path)
                              for cdata in contours]  # list of img_arr, contour_arr, im_id

        # debug: there are multiple contours for the same image indepently
        # sum contour arrays and generate new img_contour_arrays
        contour_dict = defaultdict(int)
        for im_arr, cntr_arr, im_id in img_contour_arrays:
            contour_dict[im_id] += cntr_arr
        image_dict = {}
        for im_arr, cntr_arr, im_id in img_contour_arrays:
            image_dict[im_id] = im_arr
        img_contour_arrays = [(image_dict[k], contour_dict[k], k) for k in image_dict]
        return img_contour_arrays

    def coord2pixels(self, contour_dataset, image_path):
        """
        Given a contour dataset (a DICOM class) and path that has .dcm files of
        corresponding images. This function will return img_arr and contour_arr (2d image and contour pixels)
        Inputs
            contour_dataset: DICOM dataset class that is identified as (3006, 0016)  Contour Image Sequence
            path: string that tells the path of all DICOM images
        Return
            img_arr: 2d np.array of image with pixel intensities
            contour_arr: 2d np.array of contour with 0 and 1 labels
        """

        contour_coord = contour_dataset.ContourData
        # x, y, z coordinates of the contour in mm
        coord = []
        for i in range(0, len(contour_coord), 3):
            coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))

        # extract the image id corresponding to given countour
        # read that dicom file
        fpaths = [image_path + f for f in os.listdir(image_path) if '.dcm' in f]
        print(fpaths)
        img_ID = contour_dataset.ContourImageSequence[0].ReferencedSOPInstanceUID
        for fpath in fpaths:
            img = pydicom.read_file(fpath)
            scan_ID = img.SOPInstanceUID
            print(scan_ID)
            if img_ID == scan_ID:
                print("GOT EM")
                img_arr = img.pixel_array
                break
            else:
                print("Wrong File.")
        # physical distance between the center of each pixel
        x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

        # this is the center of the upper left voxel
        origin_x, origin_y, _ = img.ImagePositionPatient

        # y, x is how it's mapped
        pixel_coords = [(np.ceil((y - origin_y) / y_spacing),
                         np.ceil((x - origin_x) / x_spacing)) for x, y, _ in coord]

        # get contour data for the image
        rows = []
        cols = []
        for i, j in list(set(pixel_coords)):
            if i < img_arr.shape[0] and j < img_arr.shape[1]:
                rows.append(i)
                cols.append(j)
            else:
                print("Out of bounds")
        print("ROWS:", rows)
        print("COLS:", cols)
        print(type(rows))
        print(img_arr.shape[0], img_arr.shape[1])
        contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8,
                                 shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

        return img_arr, contour_arr, img_ID

    def create_image_mask_files(self, patient, contour_path, image_path, index, img_format='nii'):
        """
        Create image and corresponding mask files under to folders '/images' and '/masks'
        in the parent directory of path.

        Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
            index (int): index of the desired ROISequence
            img_format (str): image format to save by, png by default
        """

        # Extract Arrays from DICOM
        X, Y = self.get_data(contour_path, image_path, index)
        Y = np.array([fill_contour(y) if y.max() == 1 else y for y in Y])

        # Create images and masks folders
        new_path = 'E:/Mphys/MaskPatients/{}'.format(patient)
        os.makedirs(new_path)
        os.makedirs(new_path + '/images/', exist_ok=True)
        os.makedirs(new_path + '/masks/', exist_ok=True)
        image_outdir = os.path.join(new_path, 'images')
        masks_outdir = os.path.join(new_path, 'masks')
        X_img = {}
        Y_img = {}
        for i in range(len(X)):
            X_img[i] = sitk.GetImageFromArray(X[i, :, :])
            Y_img[i] = sitk.GetImageFromArray(Y[i, :, :])
            sitk.WriteImage(X_img[i], '{}/image_{}.{}'.format(image_outdir, i, img_format))
            sitk.WriteImage(Y_img[i], '{}/mask_{}.{}'.format(masks_outdir, i, img_format))
