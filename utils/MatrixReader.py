from numpy import *
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
from normalization.MinMaxNormalization import MinMaxNormalization


class MatrixReader:

    def __init__(self, list_matrix_filename):
        self.list_matrix_filename = list_matrix_filename
        self.list_matrices = []
        self.list_names = []
        self.get_normalized_matrices()

    def get_normalized_matrices(self):

        # global min max for true or observed data
        min_value_to = sys.float_info.max
        max_value_to = sys.float_info.min

        # global min max for simulation data
        min_value_si = sys.float_info.max
        max_value_si = sys.float_info.min

        for line in self.list_matrix_filename:
            # print(line)
            mat_contents = sio.loadmat(line)
            # print(mat_contents)
            base_name = os.path.basename(line)
            name_matrix = base_name.split('.mat')[0]
            name_content = name_matrix
            try:
                original_matrix = mat_contents[name_content]
            except:
                name_content = base_name.split('_T')[0]
                original_matrix = mat_contents[name_content]

            matrix_size = original_matrix.shape

            if len(matrix_size) > 2:
                localmin_value_si = np.nanmin(original_matrix)
                localmax_value_si = np.nanmax(original_matrix)

                if max_value_si < localmax_value_si:
                    max_value_si = localmax_value_si
                if min_value_si > localmin_value_si:
                    min_value_si = localmin_value_si

            else:
                localmin_value_to = np.nanmin(original_matrix)
                localmax_value_to = np.nanmax(original_matrix)

                if max_value_to < localmax_value_to:
                    max_value_to = localmax_value_to
                if min_value_to > localmin_value_to:
                    min_value_to = localmin_value_to

        for line in self.list_matrix_filename:
            # print(line)
            mat_contents = sio.loadmat(line)
            # print(mat_contents)
            base_name = os.path.basename(line)
            name_matrix = base_name.split('.mat')[0]
            name_content = name_matrix
            try:
                original_matrix = mat_contents[name_content]
            except:
                name_content = base_name.split('_T')[0]
                original_matrix = mat_contents[name_content]

            matrix_size = original_matrix.shape

            # simulation matrix
            if len(matrix_size) > 2:
                # minmax normalization
                norm_matrix = MinMaxNormalization.normalize(original_matrix, min_value_si, max_value_si, 0, 1)
                norm_matrix = np.nan_to_num(norm_matrix)

                matrix_number = matrix_size[2]
                for x in range(matrix_number):
                    name_matrix_x = name_matrix + str('_') + str(x+1)
                    self.list_names.append(name_matrix_x)
                    self.list_matrices.append(norm_matrix[:, :, x])
                    # if x == 249:
                        # plt.imshow(norm_matrix[:, :, x], cmap='nipy_spectral', vmin=0, vmax=1.0,
                        #           interpolation='nearest')
                        # plt.colorbar()
                        # plt.show()

            else:

                # minmax normalization
                norm_matrix = MinMaxNormalization.normalize(original_matrix, min_value_to, max_value_to, 0, 1)
                norm_matrix = np.nan_to_num(norm_matrix)

                self.list_matrices.append(norm_matrix)
                self.list_names.append(name_matrix)

                # plt.imshow(norm_matrix, cmap='nipy_spectral', vmin=0, vmax=1.0,  interpolation='nearest')
                # plt.colorbar()
                # plt.show()
