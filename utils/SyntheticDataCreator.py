from numpy import *
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from normalization.MinMaxNormalization import MinMaxNormalization
# from scipy.ndimage import zoom
from skimage.transform import resize, rescale, downscale_local_mean
import matplotlib.pyplot as plt
from collections import Counter
import cv2


class SyntheticDataCreator:

    def __init__(self, file_name):
        mat_contents = sio.loadmat(file_name)
        # print(mat_contents)
        base_name = os.path.basename(file_name)
        name_matrix = base_name.split('.mat')[0]
        name_content = name_matrix
        try:
            self.original_matrix = mat_contents[name_content]
        except:
            name_content = base_name.split('_T')[0]
            self.original_matrix = mat_contents[name_content]

        # minmax normalization
        min_value = np.nanmin(self.original_matrix)
        max_value = np.nanmax(self.original_matrix)
        self.norm_matrix_regions = MinMaxNormalization.normalize(self.original_matrix, min_value, max_value,
                                                                 1, 10.0)
        self.norm_matrix_regions = np.nan_to_num(self.norm_matrix_regions)

        self.norm_matrix_contrast = self.norm_matrix_regions.copy()

        self.norm_matrix_size = self.norm_matrix_regions.copy()
        self.norm_matrix_size_tmp = self.norm_matrix_size.copy()

    def decrease_regions(self, value):
        matrix_size = self.norm_matrix_regions.shape
        for i in range(matrix_size[0]):
            for j in range(matrix_size[1]):
                if (self.norm_matrix_regions[i][j] > 5) and ((self.norm_matrix_regions[i][j] - value) > 5):
                    self.norm_matrix_regions[i][j] = self.norm_matrix_regions[i][j] - value
                else:
                    if (self.norm_matrix_regions[i][j] > 5) and ((self.norm_matrix_regions[i][j] - value) < 5):
                        self.norm_matrix_regions[i][j] = 5

    def generate_syntheticdata_by_regions(self, factor):
        decrease = 0.0
        for x in range(1, 21):
            self.decrease_regions(decrease)
            a = dict()
            a[str(x)] = self.norm_matrix_regions
            sio.savemat("../data/regions/"+str(x)+".mat", a)
            print(x)
            plt.imshow(self.norm_matrix_regions, cmap='nipy_spectral', vmin=0.0, vmax=10.0,  interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/regions/" + str(x) + ".png",  bbox_inches='tight', pad_inches=0, transparent=True)

            plt.show()
            decrease = decrease + factor #0.025

    def decrease_regions_automatic(self, value, detected_image):
        matrix_size = self.norm_matrix_regions.shape
        for i in range(matrix_size[0]):
            for j in range(matrix_size[1]):
                if detected_image[i][j] == 0:
                    self.norm_matrix_regions[i][j] = self.norm_matrix_regions[i][j] - value
                    if self.norm_matrix_regions[i][j] <6:
                        self.norm_matrix_regions[i][j] = 6

    def detect_image_background(self, matrix):
        h, w = matrix.shape
        manual_count = {}
        detected_image = matrix.copy()
        for y in range(0, h):
            for x in range(0, w):
                value = round(matrix[y][x])
                if value in manual_count:
                    manual_count[value] += 1
                else:
                    manual_count[value] = 1
        number_counter = Counter(manual_count).most_common(2)
        common_values = []
        for value, number in number_counter:
            common_values.append(value)

        print(common_values)
        for y in range(0, h):
            for x in range(0, w):
                if round(matrix[y][x]) in common_values:
                    detected_image[y][x] = 1
                else:
                    detected_image[y][x] = 0

        #plt.imshow(matrix, cmap='nipy_spectral', vmin=0.0, vmax=10.0, interpolation='nearest')
        # plt.colorbar()
        #plt.axis('off')
        #plt.axes().get_xaxis().set_visible(False)
        #plt.axes().get_yaxis().set_visible(False)
        #plt.show()
        return detected_image

    def generate_syntheticdata_by_regions_automatic(self, factor):
        detected_matrix = self.detect_image_background(self.norm_matrix_regions)
        decrease = 0.0
        for x in range(1, 21):
            self.decrease_regions_automatic(decrease, detected_matrix)
            a = dict()
            a[str(x)] = self.norm_matrix_regions
            sio.savemat("../data/regionsauto/"+str(x)+".mat", a)
            print(x)
            plt.imshow(self.norm_matrix_regions, cmap='nipy_spectral', vmin=0.0, vmax=10.0,  interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/regionsauto/" + str(x) + ".png",  bbox_inches='tight', pad_inches=0, transparent=True)

            plt.show()
            decrease = decrease + factor #0.025

    def decrease_contrast(self, value):
        matrix_size = self.norm_matrix_contrast.shape
        for i in range(matrix_size[0]):
            for j in range(matrix_size[1]):
                if (self.norm_matrix_contrast[i][j] > 0.0) and ((self.norm_matrix_contrast[i][j] - value) > 0.0):
                    self.norm_matrix_contrast[i][j] = self.norm_matrix_contrast[i][j] - value
                else:
                    if (self.norm_matrix_contrast[i][j] > 0.0) and ((self.norm_matrix_contrast[i][j] - value) < 0.0):
                        self.norm_matrix_contrast[i][j] = 0.0

    def generate_syntheticdata_by_contrast(self, factor):
        decr = 0.0
        for x in range(1, 21):
            self.decrease_contrast(decr)
            a = dict()
            a[str(x)] = self.norm_matrix_contrast
            sio.savemat("../data/contrast/"+str(x)+".mat", a)
            print(x)
            plt.imshow(self.norm_matrix_contrast, cmap='nipy_spectral', vmin=0.0, vmax=10.0,  interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/contrast/" + str(x) + ".png",  bbox_inches='tight', pad_inches=0, transparent=True)

            plt.show()
            decr = decr + factor #0.025

    # this method was not working
    # def decrease_size(self, value):
    #    self.norm_matrix_size_tmp = zoom(self.norm_matrix_size, (value, value))

    # def generate_syntheticdata_by_size(self):
    #    decr = 1.0
    #    for x in range(1, 11):
    #        a = {}
    #        self.decrease_size(decr)
    #        a[str(x)] = self.norm_matrix_size_tmp
    #        decr = decr - 0.1
    #        sio.savemat("../data/size/" + str(x) + ".mat", a)
    #        plt.imshow(self.norm_matrix_size_tmp, cmap='nipy_spectral', vmin=0.0, vmax=10.0, interpolation='nearest')
    #        # plt.colorbar()
    #        plt.savefig("../data/size/" + str(x) + ".png")
    #        plt.show()

    def generate_syntheticdata_by_size(self):
        matrix_size = self.norm_matrix_size_tmp.shape
        print(matrix_size)
        rows = matrix_size[0]
        cols = matrix_size[1]

        for x in range(1, 21):
            a = {}
            a[str(x)] = self.norm_matrix_size_tmp
            sio.savemat("../data/size/" + str(x) + ".mat", a)
            plt.imshow(self.norm_matrix_size_tmp, cmap='nipy_spectral', vmin=0.0, vmax=10.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/size/" + str(x) + ".png",  bbox_inches='tight', pad_inches=0, transparent=True)
            plt.show()

            print(rows, cols)
            self.norm_matrix_size_tmp = resize(self.norm_matrix_size_tmp, (rows, cols), anti_aliasing=False,
                                               order = 0, mode = 'symmetric', cval = 0, clip = True, preserve_range = False)
            rows = round(rows/1.2)
            cols = round(cols/1.2)

    def generate_syntheticdata_by_resolution(self):
        matrix_size = self.norm_matrix_size_tmp.shape
        matrix = self.norm_matrix_size_tmp
        print(matrix_size)
        rows = matrix_size[0]
        cols = matrix_size[1]
        original_rows = matrix_size[0]
        original_cols = matrix_size[1]

        for x in range(1, 21):
            a = {}
            a[str(x)] = matrix
            sio.savemat("../data/resolution/" + str(x) + ".mat", a)
            plt.imshow(matrix, cmap='nipy_spectral', vmin=0.0, vmax=10.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/resolution/" + str(x) + ".png",  bbox_inches='tight', pad_inches=0, transparent=True)
            plt.show()

            print(rows, cols)
            matrix = resize(self.norm_matrix_size_tmp, (rows, cols), anti_aliasing=False,
                                               order = 0, mode = 'symmetric', cval = 0, clip = True, preserve_range = False)
            matrix = cv2.resize(matrix, (original_cols, original_rows), interpolation=0)
            rows = round(rows/1.1)
            cols = round(cols/1.1)



def main():
    file_name = "../data/DIP_seis_obs_234x326_T4018.mat"
    synthetic_data_creator = SyntheticDataCreator(file_name)
    # synthetic_data_creator.generate_syntheticdata_by_regions(0.025)
    # synthetic_data_creator.generate_syntheticdata_by_regions_automatic(0.015)
    # synthetic_data_creator.generate_syntheticdata_by_contrast(0.012)
    # synthetic_data_creator.generate_syntheticdata_by_size()
    synthetic_data_creator.generate_syntheticdata_by_resolution()


if __name__ == "__main__":
    main()
