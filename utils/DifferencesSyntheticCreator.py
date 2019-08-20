import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from utils.MatrixReader import MatrixReader
from numpy import ones, vstack
from numpy.linalg import lstsq
import random
from scipy import interpolate


class DifferencesSyntheticCreator:
    def __init__(self, path1, path2, path3, path4):
        matrix_files = [path1, path2, path3, path4]
        self.matrixreader = MatrixReader(matrix_files)

    def calculate(self):
        matrix1 = self.matrixreader.list_matrices[0]
        matrix2 = self.matrixreader.list_matrices[1]
        matrix3 = self.matrixreader.list_matrices[2]
        matrix4 = self.matrixreader.list_matrices[3]

        mask = (matrix1 - matrix4)

        # print(str(matrix2[20][0]) +  " - " + str(matrix1[20][0]) + " -  "+ str(mask[20][0]))
        plt.imshow(mask, cmap='nipy_spectral', vmin=0, vmax=1.0, interpolation='nearest')
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/sequenced/originaldiff.png", bbox_inches='tight', pad_inches=0, transparent=True)
        # plt.colorbar()
        plt.show()

        mask = mask/20
        plt.imshow(mask, cmap='nipy_spectral', vmin=0, vmax=1.0, interpolation='nearest')
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/sequenced/reduceddiff.png", bbox_inches='tight', pad_inches=0, transparent=True)
        # plt.colorbar()
        plt.show()

        for x in range(2618, 4019, 73):
            a = dict()
            a[str(x)] = matrix1
            sio.savemat("../data/sequenced/" + str(x) + ".mat", a)
            print(x)
            plt.imshow(matrix1, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/sequenced/" + str(x) + ".png", bbox_inches='tight', pad_inches=0, transparent=True)
            plt.show()
            matrix1 = matrix1 - mask

    def calculate_linear_function(self):
        matrix1 = self.matrixreader.list_matrices[0]
        matrix2 = self.matrixreader.list_matrices[1]
        matrix3 = self.matrixreader.list_matrices[2]
        matrix4 = self.matrixreader.list_matrices[3]
        size = matrix1.shape

        # calculating from matrix1 to matrix 2

        matrixc = np.zeros([size[0], size[1]])
        matrixm = np.zeros([size[0], size[1]])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                points = [(2618, matrix1[i][j]), (3287, matrix2[i][j])]
                x_coords, y_coords = zip(*points)
                a = vstack([x_coords, ones(len(x_coords))]).T
                m, c = lstsq(a, y_coords)[0]
                matrixc[i][j] = c
                matrixm[i][j] = m

        for x in range(2618, 3276, 73):
            a = dict()

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    current_matrix[i][j] = matrixm[i][j] * x + matrixc[i][j]

            a[str(x)] = current_matrix
            sio.savemat("../data/sequencedLinear/" + str(x) + ".mat", a)
            print(x)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/sequencedLinear/" + str(x) + ".png", bbox_inches='tight', pad_inches=0, transparent=True)
            plt.show()

        # calculating from matrix2 to matrix 3

        matrixc = np.zeros([size[0], size[1]])
        matrixm = np.zeros([size[0], size[1]])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                points = [(3287, matrix2[i][j]), (3652, matrix3[i][j])]
                x_coords, y_coords = zip(*points)
                a = vstack([x_coords, ones(len(x_coords))]).T
                m, c = lstsq(a, y_coords)[0]
                matrixc[i][j] = c
                matrixm[i][j] = m

        for x in range(3348, 3641, 73):
            a = dict()

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    current_matrix[i][j] = matrixm[i][j] * x + matrixc[i][j]

            a[str(x)] = current_matrix
            sio.savemat("../data/sequencedLinear/" + str(x) + ".mat", a)
            print(x)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/sequencedLinear/" + str(x) + ".png", bbox_inches='tight', pad_inches=0, transparent=True)
            plt.show()

        # calculating from matrix3 to matrix4

        matrixc = np.zeros([size[0], size[1]])
        matrixm = np.zeros([size[0], size[1]])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                points = [(3652, matrix3[i][j]), (4018, matrix4[i][j])]
                x_coords, y_coords = zip(*points)
                a = vstack([x_coords, ones(len(x_coords))]).T
                m, c = lstsq(a, y_coords)[0]
                matrixc[i][j] = c
                matrixm[i][j] = m

        for x in range(3713, 4019, 73):
            a = dict()

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    current_matrix[i][j] = matrixm[i][j] * x + matrixc[i][j]

            a[str(x)] = current_matrix
            sio.savemat("../data/sequencedLinear/" + str(x) + ".mat", a)
            print(x)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/sequencedLinear/" + str(x) + ".png", bbox_inches='tight', pad_inches=0,
                        transparent=True)
            plt.show()

    def calculate_value_degree(self, x, coefficients):
        value = 0
        size_coeff = len(coefficients)
        for n in range(size_coeff, 0, -1):
            value += coefficients[size_coeff - n - 1] * pow(x, n - 1)
        return value

    def calculate_polynomial_function(self, degree):
        matrix1 = self.matrixreader.list_matrices[0]
        matrix2 = self.matrixreader.list_matrices[1]
        matrix3 = self.matrixreader.list_matrices[2]
        matrix4 = self.matrixreader.list_matrices[3]
        size = matrix1.shape

        for x in range(2618, 4019, 73):
            a = dict()

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 3652, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix3[i][j], matrix4[i][j]]
                    coeffs = np.polyfit(xs, ys, degree)
                    ffit = np.poly1d(coeffs)
                    current_matrix[i][j] = round(float(ffit(x)), 4)

            a[str(x)] = current_matrix
            sio.savemat("../data/sequencedPolynomial/" + str(degree) + "/" + str(x) + ".mat", a)
            print(x)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/sequencedPolynomial/"+ str(degree) + "/"  + str(x) + ".png", bbox_inches='tight', pad_inches=0,
                        transparent=True)
            plt.show()

    def calculate_cubic_spline_function(self):
        matrix1 = self.matrixreader.list_matrices[0]
        matrix2 = self.matrixreader.list_matrices[1]
        matrix3 = self.matrixreader.list_matrices[2]
        matrix4 = self.matrixreader.list_matrices[3]
        size = matrix1.shape

        for x in range(2618, 4019, 73):
            a = dict()

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 3652, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix3[i][j], matrix4[i][j]]
                    # cs = CubicSpline(xs, ys)
                    cs = interpolate.splrep(xs, ys)
                    #current_matrix[i][j] = round(float(cs(x)), 4)
                    current_matrix[i][j] = round(float(interpolate.splev(x, cs)), 4)
                    

            a[str(x)] = current_matrix
            sio.savemat("../data/sequencedCubic/" + str(x) + ".mat", a)
            print(x)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/sequencedCubic/" + str(x) + ".png", bbox_inches='tight', pad_inches=0,
                        transparent=True)
            plt.show()

    def calculate_piecewise_linear_function(self):

        matrix1 = self.matrixreader.list_matrices[0]
        matrix2 = self.matrixreader.list_matrices[1]
        matrix3 = self.matrixreader.list_matrices[2]
        matrix4 = self.matrixreader.list_matrices[3]
        size = matrix1.shape

        for x in range(2618, 4019, 73):
            a = dict()

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 3652, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix3[i][j], matrix4[i][j]]
                    current_matrix[i][j] = round(float(np.interp(x, xs, ys)), 4)

            a[str(x)] = current_matrix
            sio.savemat("../data/sequencedPieceWise/" + str(x) + ".mat", a)
            print(x)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/sequencedPieceWise/" + str(x) + ".png", bbox_inches='tight', pad_inches=0,
                        transparent=True)
            plt.show()

    def calculate_piecewise_linear_function_with_real(self):

        matrix1 = self.matrixreader.list_matrices[0]
        matrix2 = self.matrixreader.list_matrices[1]
        matrix3 = self.matrixreader.list_matrices[2]
        matrix4 = self.matrixreader.list_matrices[3]
        size = matrix1.shape

        name_list = list()
        time_list = list()
        time_list.append(2618)
        name_list.append("1Y-0")

        time_list.append(2713)
        time_list.append(2808)
        time_list.append(2903)
        time_list.append(2998)
        time_list.append(3093)
        time_list.append(3188)
        name_list.append("1Y-1")
        name_list.append("1Y-2")
        name_list.append("1Y-3")
        name_list.append("1Y-4")
        name_list.append("1Y-5")
        name_list.append("1Y-6")

        time_list.append(3287)
        name_list.append("2Y-0")

        time_list.append(3348)
        time_list.append(3409)
        time_list.append(3470)
        time_list.append(3531)
        time_list.append(3592)
        name_list.append("2Y-1")
        name_list.append("2Y-2")
        name_list.append("2Y-3")
        name_list.append("2Y-4")
        name_list.append("2Y-5")

        time_list.append(3652)
        name_list.append("3Y-0")

        time_list.append(3713)
        time_list.append(3774)
        time_list.append(3835)
        time_list.append(3896)
        time_list.append(3957)
        name_list.append("3Y-1")
        name_list.append("3Y-2")
        name_list.append("3Y-3")
        name_list.append("3Y-4")
        name_list.append("3Y-5")

        time_list.append(4018)
        name_list.append("4Y-0")

        for index in range(0, len(time_list)):
            a = dict()
            x = time_list[index]
            name = name_list[index]

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 3652, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix3[i][j], matrix4[i][j]]
                    current_matrix[i][j] = round(float(np.interp(x, xs, ys)), 4)

            a[str(name)] = current_matrix
            sio.savemat("../data/sequencedPieceWiseReal/" + name + ".mat", a)
            print(name)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/sequencedPieceWiseReal/" + name + ".png", bbox_inches='tight', pad_inches=0,
                        transparent=True)
            plt.show()
            
    def calculate_gaussian_noise(self):
        path = "/home/aurea/Documents/Shell4DSimilarity/data/interpolated_klaus_data/dados_aurea/no_uncertainties/T2618_linear_transform_model_99.mat"
        pregaussian_files = [path]
        matrixreader = MatrixReader(pregaussian_files)
        
        
        
        for index in range(0, len(matrixreader.list_matrices)):        
            matrix = matrixreader.list_matrices[index]
            a = dict()
            mu, sigma = 0, round(random.uniform(0, 0.01),4)
            print(sigma)
            noise = np.random.normal(mu, sigma, matrix.shape) 
            matrix = matrix + noise
            name = matrixreader.list_names[index]
            a[str(name)] = matrix
            sio.savemat("../data/sequencedGaussianNoise/" + str(name) + ".mat", a)
            print(name)
            plt.imshow(matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/sequencedGaussianNoise/" + str(name) + ".png", bbox_inches='tight', pad_inches=0,
                       transparent=True)
            plt.show()

    def calculate_basic_rectangle(self):

        matrix1 = [[0, 0, 0, 0, 0, 0], 
                   [0, 1, 1, 1, 1, 0], 
                   [0, 1, 1, 1, 1, 0], 
                   [0, 1, 1, 1, 1, 0], 
                   [0, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0]]
        
        a = dict()
        # calculating matrix
        # current_matrix = np.zeros([size[0], size[1]])
        a["rectangle"] = matrix1
        sio.savemat("../data/rectangle.mat", a)
        print("rectangle")
        plt.imshow(matrix1, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/rectangle.png", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()

def main():
    path1 = "/home/aurea/Documents/Shell4DSimilarity/data/DIP_seis_obs_234x326_T2618.mat"
    path2 = "/home/aurea/Documents/Shell4DSimilarity/data/DIP_seis_obs_234x326_T3287.mat"
    path3 = "/home/aurea/Documents/Shell4DSimilarity/data/DIP_seis_obs_234x326_T3652.mat"
    path4 = "/home/aurea/Documents/Shell4DSimilarity/data/DIP_seis_obs_234x326_T4018.mat"
    calculator = DifferencesSyntheticCreator(path1, path2, path3, path4)

    # calculator.calculate()
    # calculator.calculate_linear_function()
    # calculator.calculate_cubic_spline_function()
    # calculator.calculate_piecewise_linear_function()
    # calculator.calculate_piecewise_linear_function_with_real()
    # calculator.calculate_polynomial_function(1)
    # calculator.calculate_polynomial_function(2)
    # calculator.calculate_polynomial_function(3)
    # calculator.calculate_polynomial_function(4)
    # calculator.calculate_polynomial_function(5)
    # calculator.calculate_gaussian_noise()

    # coeff = np.polyfit([1, 2, 3, 4], [0, -1, -2, -3], 2)
    # ffit = np.poly1d(coeff)
    # print(ffit)
    # print(round(float(ffit(3)), 4))

    # cs = CubicSpline([1, 2, 3, 4], [0, -1, -2, -3])
    # print(round(float(cs(5)), 4))
    calculator.calculate_basic_rectangle()


if __name__ == "__main__":
    main()






