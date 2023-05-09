#!/usr/bin/python3.7
# coding=utf-8

"""
Python file defining classes for custom Image objects
"""
import os.path
import sys
import csv
import numpy as np
import pandas as pd
from skimage import filters
from skimage import morphology
from skimage import img_as_float, img_as_uint
from skimage import io
from skimage import restoration

from pystackreg.util import to_uint16

__author__ = "Altair C. Hernandez"
__copyright__ = 'Copyright 2022, The Exocystosis Modeling Project'
__credits__ = ["Oriol Gallego", "Radovan Dojcilovic", "Andrea Picco", "Damien P. Devos"]
__version__ = "2.0"
__maintainer__ = "Altair C. Hernandez"
__email__ = "altair.chinchilla@upf.edu"
__status__ = "Development"


class BioImage:
    """
    Class for images coming from PICT (Protein Interaction
    from imaged Complexes after Translocation) experiments.

    Attributes:
        image_path = path_to_image/image_id.tif
        Id = id from image_path
        raw_name = name of raw image
        bgn_name = name for Background Subtracted image using
         scikit-image using Rolling Ball algorithm
        md_name = name for media filtered image using
        scikit-image using median filter algorithm
        bgn_md_name = name for Background Subtracted and Median filtered
         image.
        raw_npimage = ndarray
    """

    def __init__(self, image_path):
        self.image_path = image_path
        self._image_bgn = np.zeros(shape=self.image.shape, dtype=self.image.dtype)
        self._image_bgn_md = np.zeros(shape=self.image.shape, dtype=self.image.dtype)

    @property
    def Id(self):
        """"""
        return self.image_path.split("/")[-1].split(".")[0].split("_")[-1]

    @property
    def name(self):
        """"""
        return self.image_path.split("/")[-1].split(".")[0]

    @property
    def bgn_name(self):
        """"""
        return "image_{}".format(self.Id)

    @property
    def bgn_median_name(self):
        """"""
        return "imageMD_{}".format(self.Id)

    @property
    def max(self):
        """Return max of intensity of raw_image"""
        return self.image.max()

    @property
    def min(self):
        """Return min of intensity of raw_image"""
        return self.image.min()

    @property
    def image(self):
        """Get raw image array"""
        return io.imread(self.image_path)

    def remove_frames(self, frames_to_remove):
        """"""
        image = self.image.copy()
        print(f"image shape is {image.shape}")
        processed_image = np.zeros(shape=self.image.shape, dtype=self.image.dtype)
        for f in range(image.shape[2]):
            print(f)
            if f not in frames_to_remove:  # exclude frames in list
                print("yes ")
                processed_image[:, :, f] += image[:, :, f]
        print(f"processed image shape is {processed_image.shape}")
        exit()
        return processed_image

    def subtract_background(self, path_to_save, radius, kernel=None):
            """
            Method to perform background subtraction
            using rolling ball radius
            Parameters
            ----------
            path_to_save: path to save resulting image background subtracted
            kernel: The kernel to be rolled/translated in the image.
            It must have the same number of dimensions as the image.
            radius: radius to compute the kernel for rolling ball

            Return
            -------
            Image BGN subtracted as ndarray
            """
            img_bgn = np.zeros(shape=self.image.shape, dtype=self.image.dtype)
            norm_radius = radius / self.image.max()  # normalize the radius according to image.max value
            # Define the kernel dimensions and intensity. Kernel dim should be == to image dim
            kernel = restoration.ellipsoid_kernel((radius * 2, radius * 2), norm_radius)
            sys.stdout.write("Background Subtraction on {}\n".format(self.name))
            # Iterate through frames
            for f in range(self.image.shape[0]):
                # compute the background of the cell for the frame f
                img_bgn[f, :, :] += restoration.rolling_ball(self.image[f, :, :], kernel=kernel)
                self._image_bgn[f, :, :] = self.image[f, :, :] - img_bgn[f, :, :]

            # Save tif files
            if not os.path.isdir(path_to_save):
                os.mkdir(path_to_save)
            # Save image Background Subtracted (BGN)
            io.imsave(path_to_save + self.bgn_name + ".tif", self._image_bgn, plugin="tifffile",
                      check_contrast=False)
            print("\t{} saved in {}\n".format(self.name, path_to_save))
            return self

    def median_filter(self, path_to_save, median_radius):
        """
        Method to perform median filter from an image,
        computing the image corrected without cytoplasmatic
         background.

        Parameters
        ----------
        path_to_save(string): path to save resulting image background subtracted
        median_radius(int): radius to compute the median filter (default:10)

        Returns
        ----------
        img_corrected, img_median
        """
        img_median = np.zeros(shape=self.image.shape, dtype=self.image.dtype)
        sys.stdout.write("Background Subtraction and Median Filter on {}\n".format(self.name))
        # Iterate through frames in image
        for f in range(self.image.shape[0]):
            # compute the median of the cell for the frame f
            img_median[f, :, :] = filters.median(self.image[f, :, :], morphology.disk(median_radius))

            # compute the image corrected without cytoplasmatic background.
            # All maths are done on signed float dtype and converted in 'unsinged 16 bit' format
            self._image_bgn_md[f, :, :] = img_as_uint(
                (img_as_float(self.image[f, :, :]) - img_as_float(img_median[f, :, :]))
            )
        # Save tif files
        if not os.path.isdir(path_to_save):
            os.mkdir(path_to_save)
        # Save image BGN-MD
        io.imsave(path_to_save + self.bgn_median_name + ".tif", self._image_bgn_md, plugin="tifffile",
                  check_contrast=False)
        sys.stdout.write("{} saved in {}\n".format(self.bgn_median_name, path_to_save))
        return self

    def do_warping(self, sr_object, path_to_save_warped, save_stack=True):
        """
        Apply StackReg object to do transformation on BioImages instances.

        The transformation output of pystackreg is exactly equivalent to that of the ImageJ plugins
        TurboReg/StackReg on which it is based. The output of the transform function therefore has
        a float datatype and may contain negative values. To again create an image with integer values,
        the utility function pystackreg.util.to_uint16() can be used.
        Parameters
        ----------
        sr_object (StackReg instance): StackReg object with registered beads,
         ready for applying transformation on sample images.
        path_to_save_warped (string): path to save warped images
        save_stack (bool): if True, it save [W1_warped,W2] stacks
        """
        sys.stdout.write("Doing Warping..\n")
        # Iterate through MD images and apply transformation to red channel (frame 0)
        W1 = self._image_bgn_md[0, :, :]  # Red chanel (frame 0)
        W2 = self._image_bgn_md[1, :, :]  # Green chanel (frame 1)
        # Test Swapping Channels
        # W1 = self._image_bgn_md[1, :, :]  # Green chanel (frame 1)
        # W2 = self._image_bgn_md[0, :, :]  # Red chanel (frame 0)

        # Transform Red channel with Transformation matrix in "sr_object", and
        # convert to uint16, since the transformation returns float64 with
        # negative values. The resulting W1_warped should have min = 0 and be uint16
        W1_warped = to_uint16(sr_object.transform(W1))
        # W2_warped = to_uint16(sr_object.transform(W2))
        sys.stdout.write("W1_warped generated! Saving..\n")
        # Save separate channels (W1_warped, W1, W2)
        sys.stdout.write("\tSaving W1_warped, W1 and W2\n")
        if not os.path.exists(path_to_save_warped):
            os.mkdir(path_to_save_warped)
        io.imsave(path_to_save_warped + self.bgn_median_name + "_W1_warped.tif", W1_warped, plugin="tifffile",
                  check_contrast=False)
        io.imsave(path_to_save_warped + self.bgn_median_name + "_W1.tif", W1, plugin="tifffile", check_contrast=False)
        io.imsave(path_to_save_warped + self.bgn_median_name + "_W2.tif", W2, plugin="tifffile", check_contrast=False)
        # io.imsave(path_to_save_warped + self.bgn_median_name + "_W1.tif", W2, plugin="tifffile", check_contrast=False)
        # io.imsave(path_to_save_warped + self.bgn_median_name + "_W2.tif", W1, plugin="tifffile", check_contrast=False)

        # Save stack = [W1_warped, W2] if option is True
        if save_stack:
            stack = np.stack([W1_warped, W2])
            # stack = np.stack([W1_warped, W2])
            if not os.path.isdir(path_to_save_warped + "stacks"):
                os.mkdir(path_to_save_warped + "stacks")
            io.imsave(path_to_save_warped + 'stacks/warped_stack_{}.tif'.format(self.bgn_median_name), stack,
                      plugin="tifffile",
                      check_contrast=False)
            sys.stdout.write("\twarped_stack_{}.tif saved!\n".format(self.bgn_median_name))

    @image.setter
    def image(self, value):
        self._image = value


class Biodata:
    """
    Class defined for data derived from a Bioimage instance.
    """
    sniffer = csv.Sniffer()

    def __init__(self, path):
        """
        path: data path from source dir
        """
        self.data_path = path

    @property
    def Id(self):
        """"""
        return self.data_path.split("/")[-1].split("_")[2]

    @property
    def name(self):
        """"""
        return self.data_path.split("/")[-1].split(".")[0]

    @property
    def channel(self):
        """Get channel: W1, W1_warped, W2"""
        if self.name.endswith("warped"):
            return "_".join([self.name.split("_")[-2], self.name.split("_")[-1]])  # W1_warped
        else:
            return self.name.split("_")[-1]  # W1 or W2

    @property
    def sep(self):
        """Get sep from csv file"""
        with open(self.data_path, 'r') as f:
            dialect = csv.Sniffer().sniff(f.read(30))
            return dialect.delimiter

    @property
    def data(self):
        """Read data from file"""
        if self.data_path.split("/")[-2] in ["ref", "particle_tracker"] or \
                self.data_path.split("/")[-3] in ["ref", "particle_tracker"]:
            col_names = ['y', 'x', 'm0', 'm2', 'np', 'm1', 'm3', 'm4', 'm5', 'm11', 'm20', 'm02']
            return pd.read_csv(self.data_path, sep=self.sep, names=col_names)
        elif self.data_path.split("/")[-2] == "trackpy" or self.data_path.split("/")[-3] == "trackpy":
            col_names = ['y', 'x', 'm0', 'm2', 'ecc', 'm1', 'm3', 'ori', 'ori2', 'm11', 'm20', 'm02']
            return pd.read_csv(self.data_path, sep=self.sep, names=col_names).sort_values(by='y', ascending=True)

    @property
    def image_process_type(self):
        """
        Return if data comes from a "warped" image or
        from an MD image.
        """
        if self.channel.endswith("warped"):
            return "warped"
        else:
            return "MD"

    def get_data(self, columns=None, names=None):
        """
        Read data and select columns by index
        Parameters
        ----------
        columns: list of indexes to select from data df
        names: (optional) dictionary with {original_name: name_to_change}

        Returns
        -------
        new dataframe with selected columns and names.
        """
        if columns is not None:
            if names is not None:
                return self.data.iloc[:, columns].rename(columns=names)

            else:
                return self.data.iloc[:, columns]
        else:
            return self.data

    @property
    def sample_path(self):
        """Return the path for the sample working directory (1103, 1110, ...)"""
        return "/".join(list(self.data_path.split("/"))[:-2])

    @property
    def parent_image(self):
        """
        Return parent BioImage instance from which the data have been created.
        """
        if self.image_process_type == "warped":
            return BioImage(self.sample_path + "/output/warped/imageMD_{}_{}.tif".format(self.Id, self.channel))
        elif self.image_process_type == "MD":
            return BioImage(self.sample_path + "/output/images/imageMD_{}.tif".format(self.Id))


if __name__ == "__main__":
    print("Custom Classes\n"
          "\t- BioImage\n"
          "\t- Biodata\n")
    sys.exit(0)
