import gdcm
import numpy
import os
from operator import itemgetter
import json
import math
import operator
from scipy import ndimage

class DicomImageTypeNotSupported(Exception):
    pass


GDCM_TO_NUMPY = {
    gdcm.PixelFormat.UINT8: numpy.uint8,
    gdcm.PixelFormat.INT8: numpy.int8,
    gdcm.PixelFormat.UINT16: numpy.uint16,
    gdcm.PixelFormat.INT16: numpy.int16,
    gdcm.PixelFormat.UINT32: numpy.uint32,
    gdcm.PixelFormat.INT32: numpy.int32,
    gdcm.PixelFormat.FLOAT32: numpy.float32,
    gdcm.PixelFormat.FLOAT64: numpy.float64
}


def get_numpy_array_type(gdcm_pixel_format):
    return GDCM_TO_NUMPY[gdcm_pixel_format]


def gdcm_to_array(image):
    """Convert a GDCM image to a numpy array."""
    pf = image.GetPixelFormat()

    if pf.GetScalarType() not in GDCM_TO_NUMPY:
        raise DicomImageTypeNotSupported(
            "{} image loading not supported.".format(pf.GetScalarType())
        )

    shape = (image.GetDimension(0), image.GetDimension(1))
    # if we are not single value per pixel, make shape have an array value
    if pf.GetSamplesPerPixel() > 1:
        shape = (shape[0], shape[1], pf.GetSamplesPerPixel)
    dtype = get_numpy_array_type(pf.GetScalarType())
    gdcm_array = image.GetBuffer()
    result = numpy.frombuffer(gdcm_array, dtype=dtype)
    result.shape = shape
    return result


class DicomImageReadFailed(Exception):
    pass


class DicomFileReadFailed(Exception):
    pass


class DicomFile(object):

    def __init__(self, filename):
        self.image_reader = gdcm.ImageReader()
        self.image_reader.SetFileName(filename)
        self.reader = gdcm.Reader()
        self.reader.SetFileName(filename)
        if not self.reader.Read():
            raise DicomFileReadFailed("{} dicom file read failed.".format(
                filename))
        self.dicom_file = self.reader.GetFile()
        self.dicom_dataset = self.dicom_file.GetDataSet()
        self.dicom_header = self.dicom_file.GetHeader()
        if not self.image_reader.Read():
            raise DicomImageReadFailed("{} image read failed.".format(
                filename))

    def is_overlay_outdated(self):
        if self.dicom_dataset.FindDataElement(gdcm.Tag(0x6000, 0x0100)):
            return self.get_python_filter().ToPyObject(
                gdcm.Tag(0x6000, 0x0100)) == 16
        else:
            return False

    def get_python_filter(self):
        f = gdcm.PythonFilter()
        f.SetFile(self.dicom_file)
        return f

    def get_instance_number(self):
        return int(str(self.dicom_dataset.GetDataElement(
            gdcm.Tag(0x00020, 0x0013)).GetValue()))

    def get_numpy_pixel_array(self):
        return gdcm_to_array(self.image_reader.GetImage())

    def get_image_orientation(self):
        return [float(x) for x in str(self.dicom_dataset.GetDataElement(
            gdcm.Tag(0x00020, 0x0037)).GetValue()).split("\\")]

    def get_image_orientation_row(self):
        return self.get_image_orientation()[0:3]

    def get_image_orieintation_column(self):
        return self.get_image_orientation()[3:]

    def get_image_position(self):
        return [float(x) for x in str(self.dicom_dataset.GetDataElement(
            gdcm.Tag(0x00020, 0x0032)).GetValue()).split("\\")]

    def get_accession_number(self):
        return str(self.dicom_dataset.GetDataElement(
            gdcm.Tag(0x0008, 0x0050)).GetValue())

    def get_pixel_spacing(self):
        return [float(x) for x in str(self.dicom_dataset.GetDataElement(
            gdcm.Tag(0x00028, 0x0030)).GetValue()).split("\\")]

    def get_row_spacing(self):
        return self.get_pixel_spacing()[0]

    def get_column_spacing(self):
        return self.get_pixel_spacing()[1]

    def get_series_uid(self):
        return str(self.dicom_dataset.GetDataElement(
            gdcm.Tag(0x00020, 0x000E)).GetValue())


def get_matrix_for_pixel(i, j):
    return numpy.matrix([[i], [j], [0], [1]])


def get_bounds_for_dicom_files(filedir):
    bounds = []
    fail_count = 0
    files_found = 0
    failures = []
    outdated = []
    summary_file = os.path.join(filedir, 'summary.json')
    if os.path.isfile(summary_file):
        return get_results_from_summary(summary_file)
    for filename in os.listdir(filedir):
        if not filename.endswith('.dcm'):
            continue
        files_found += 1
        rfile = os.path.join(filedir, filename)
        try:
            d = DicomFile(rfile)
        except Exception as e:
            fail_count += 1
            print(e)
            failures.append(rfile)
            continue

        if d.is_overlay_outdated():
            outdated.append(rfile)
        bounds.append(d.get_transformed_points_bounds())
    max_x = max(bounds, key=itemgetter(0))[0]
    min_x = min(bounds, key=itemgetter(0))[0]
    max_y = max(bounds, key=itemgetter(1))[1]
    min_y = min(bounds, key=itemgetter(1))[1]
    max_z = max(bounds, key=itemgetter(2))[2]
    min_z = min(bounds, key=itemgetter(2))[2]
    bounds_arr = (min_x, max_x, min_y, max_y, min_z, max_z)
    result = DicomDirectoryResults(
        bounds_arr, files_found, fail_count,
        failures, outdated
    )
    return result


def get_voxel_array_for_dicom_files(files_to_use):
    points = []
    for filename in files_to_use:
        if not filename.endswith('.dcm'):
            continue
        try:
            d = DicomFile(filename)
        except Exception as e:
            print(e)
            continue
        point_cloud = d.get_point_cloud()
        points += point_cloud
    return points


def get_results_from_summary(filename):
    with open(filename, 'r') as json_data:
        data = json.load(json_data)
        return DicomDirectoryResults(
            data[u'bounds_array'],
            data[u'files_found'],
            data[u'fail_count'],
            data[u'failures'],
            data[u'outdated']
        )


class DicomDirectoryResults(object):

    def __init__(self, bounds_array, files_found, fail_count, failures,
                 outdated):
        self.bounds_array = bounds_array
        self.files_found = files_found
        self.fail_count = fail_count
        self.failures = failures
        self.outdated = outdated

    def get_json(self):
        return {
            'bounds_array': self.bounds_array,
            'files_found': self.files_found,
            'fail_count': self.fail_count,
            'failures': self.failures,
            'outdated': self.outdated,
        }

    def write_to_json(self, filedir):
        with open(os.path.join(filedir, 'summary.json'), 'w') as outfile:
            json.dump(self.get_json(), outfile)


def load_only_summaries(start_dir):
    results = []
    for filename in os.listdir(start_dir):
        rpath = os.path.join(start_dir, filename)
        if os.path.isdir(rpath):
            summary_file = os.path.join(rpath, 'summary.json')
            if os.path.isfile(summary_file):
                results.append(get_results_from_summary(summary_file))
    return results


def get_files_grouped_by_series(file_dir):
    uids = {}
    for filename in os.listdir(file_dir):
        if not filename.endswith('.dcm'):
            continue
        rfile = os.path.join(file_dir, filename)
        try:
            d = DicomFile(rfile)
        except Exception as e:
            print(e)
            continue
        uid = d.get_series_uid()
        if uid not in uids:
            uids[uid] = [rfile]
        else:
            uids[uid].append(rfile)
    return uids

def sort_series(series_group):
    dicom_files = [DicomFile(x) for x in series_group]
    return sorted(dicom_files, key=lambda d: d.get_instance_number())


def get_array_for_series_group(series_group):
    sorted_series = sort_series(series_group)
    ret_array = None
    for index, dicom_file in enumerate(sorted_series):
        pix_array = dicom_file.get_numpy_pixel_array()
        shape = None
        if index == 0:
            shape = pix_array.shape
            new_shape = len(sorted_series), shape[0], shape[1]
            ret_array = numpy.zeros(new_shape, dtype=pix_array.dtype)
        if pix_array.shape == shape:
            ret_array[index] = pix_array
    return ret_array

MINIMUM_SERIES_SIZE = 10

def get_arrays_for_dicom_dir(start_dir):
    arrays = []
    results = get_files_grouped_by_series(start_dir)
    for series_uid in results:
        if (len(results[series_uid]) >= MINIMUM_SERIES_SIZE):
            ret_array = get_array_for_series_group(results[series_uid])
            arrays.append((series_uid, ret_array))
        else:
            print('skipping {} because series size is below minimum'.format(series_uid))
    return arrays

def resample_voxel_arrays(voxel_arrays, desired_shape):
    resampled = []
    for uid, vox_array in voxel_arrays:
        zoom_factor = tuple(map(operator.truediv, desired_shape, vox_array.shape))
        # we might not want to use bilinear interpolation, not certain
        new_array = ndimage.interpolation.zoom(vox_array, zoom_factor)
        resampled.append((uid, new_array))
    return resampled


if __name__ == '__main__':
    import os
    import sys
    from matplotlib import pyplot as plt
    import time

    start_dir = os.path.expanduser(sys.argv[1])
    voxel_arrays = get_arrays_for_dicom_dir(start_dir)
    resampled = resample_voxel_arrays(voxel_arrays, (32, 512, 512))
    for vox_array in resampled:
        plt.ion()
        print('notification that you are starting a new image')
        for frame in vox_array:
            plt.imshow(frame)
            plt.gray()
            plt.show()
            raw_input('Press enter to advance frame...')
            plt.close()
