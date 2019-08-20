import scipy.io as sio
import os
import numpy as np
from normalization.MinMaxNormalization import MinMaxNormalization
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


class MatlabMatrix:

    def __init__(self):
        self.file_names = ""
        self.list_matrices = []
        self.list_labels = []
        self.list_ids = []
        self.list_representations = []
        self.max_value = float("-inf")
        self.min_value = float("inf")
        self.matrix_size = [0, 0]

    def load(self, file_names, normalization_flag):
        self.file_names = file_names
        for line in self.file_names:
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

            self.matrix_size = original_matrix.shape
            
            mean = np.mean(original_matrix)
            std_value = np.std(original_matrix)
            
            # normalizing
            if normalization_flag == True:
                original_matrix = (original_matrix-mean)/std_value

            self.min_value = np.nanmin(original_matrix)
            self.max_value = np.nanmax(original_matrix)

            # simulation matrix
            if len(self.matrix_size) > 2:
                # minmax normalization
                # norm_matrix = MinMaxNormalization.normalize(original_matrix, self.min_value, self.max_value, 0, 1)
                norm_matrix = np.nan_to_num(original_matrix)#(norm_matrix)
                matrix_number = self.matrix_size[2]
                for x in range(matrix_number):
                    name_matrix_x = name_matrix + "_" + str(x + 1)
                    self.list_labels.append(name_matrix_x)
                    self.list_ids.append(len(self.list_ids)+1)
                    self.list_matrices.append(norm_matrix[:, :, x])
                    #plt.imshow(norm_matrix[:, :, x], cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
                    ##plt.colorbar()
                    #plt.axis('off')
                    #plt.axes().get_xaxis().set_visible(False)
                    #plt.axes().get_yaxis().set_visible(False)
                    #plt.savefig("initial_final/final/"+name_matrix_x+".png",
                    #            bbox_inches='tight', pad_inches=0, transparent=True)
                    #plt.show()

            else:
                # minmax normalization
                #norm_matrix = MinMaxNormalization.normalize(original_matrix, self.min_value, self.max_value, 0, 1)
                norm_matrix = np.nan_to_num(original_matrix)#(norm_matrix)
                self.list_matrices.append(norm_matrix)
                self.list_labels.append(name_matrix)
                self.list_ids.append(len(self.list_ids) + 1)
                #plt.imshow(norm_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
                ##plt.colorbar()
                #plt.axis('off')
                #plt.axes().get_xaxis().set_visible(False)
                #plt.axes().get_yaxis().set_visible(False)
                #plt.savefig("initial_final/final/"+name_matrix+".png",
                #            bbox_inches='tight', pad_inches=0, transparent=True)
                #plt.show()
                
                
    
    def standardize_data(self, data):
        values = data[~np.isnan(data)]
        std_values = StandardScaler().fit_transform(values.reshape(-1, 1)).reshape(-1, )
        return std_values

    def save(self, file_directory):
        for i in (0, len(self.list_matrices)):
            a = dict()
            a[str(self.list_labels[i])] = self.list_matrices[i]
            sio.savemat(file_directory + "/" + self.list_labels[i] + ".mat", a)
