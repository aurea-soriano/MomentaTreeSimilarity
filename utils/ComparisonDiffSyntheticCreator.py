import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from utils.MatrixReader import MatrixReader
from numpy import ones, vstack
from numpy.linalg import lstsq
from scipy.interpolate import CubicSpline, lagrange, NearestNDInterpolator
import os


class ComparisonDifferencesSyntheticCreator:
    def __init__(self, path1, path2, path3, path4, diff1, diff2, diff3):
        matrix_files = [path1, path2, path3, path4]
        self.diff1 = diff1
        self.diff2 = diff2
        self.diff3 = diff3
        self.matrixreader = MatrixReader(matrix_files)

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

        a = dict()
        a[str(2618)] = matrix1
        sio.savemat("../data/comparison/sequencedLinear/" + str(2618) + ".mat", a)
        print(2618)
        plt.imshow(matrix1, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedLinear/" + str(2618) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        step = (3287 - 2618) // 9
        init_count = 2618

        for x in range(2, 10):
            init_count = init_count + step
            a = dict()

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    current_matrix[i][j] = matrixm[i][j] * init_count + matrixc[i][j] # x

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedLinear/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedLinear/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        a = dict()
        a[str(3287)] = matrix2
        sio.savemat("../data/comparison/sequencedLinear/" + str(3287) + ".mat", a)
        print(3287)
        plt.imshow(matrix2, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedLinear/" + str(3287) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        # calculating from matrix2 to matrix 4

        matrixc = np.zeros([size[0], size[1]])
        matrixm = np.zeros([size[0], size[1]])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                points = [(3287, matrix2[i][j]), (4018, matrix4[i][j])]
                x_coords, y_coords = zip(*points)
                a = vstack([x_coords, ones(len(x_coords))]).T
                m, c = lstsq(a, y_coords)[0]
                matrixc[i][j] = c
                matrixm[i][j] = m

        step = (4018 - 3287) // 9
        init_count = 3287

        for x in range(11, 20):
            init_count = init_count + step
            a = dict()

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    current_matrix[i][j] = matrixm[i][j] * init_count + matrixc[i][j] # x

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedLinear/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedLinear/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                current_matrix[i][j] = matrixm[i][j] * 3652 + matrixc[i][j]  # x

        a[str(3652)] = current_matrix
        sio.savemat("../data/comparison/sequencedLinear/" + str(3652) + ".mat", a)
        print(3652)
        plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedLinear/" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a["real"+str(3652)] = matrix3
        sio.savemat("../data/comparison/sequencedLinear/real" + str(3652) + ".mat", a)
        print("real3652")
        plt.imshow(matrix3, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedLinear/real" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a[str(4018)] = matrix4
        sio.savemat("../data/comparison/sequencedLinear/" + str(4018) + ".mat", a)
        print(4018)
        plt.imshow(matrix4, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedLinear/" + str(4018) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

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

        # calculating from matrix1 to matrix 2

        a = dict()
        a[str(2618)] = matrix1
        sio.savemat("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(2618) + ".mat", a)
        print(2618)
        plt.imshow(matrix1, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(2618) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        step = (3287 - 2618) // 9
        init_count = 2618

        # calculating from matrix1 to matrix 2

        for x in range(2, 10):
            a = dict()
            init_count = init_count + step
            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                    coeffs = np.polyfit(xs, ys, degree)
                    ffit = np.poly1d(coeffs)
                    current_matrix[i][j] = round(float(ffit(init_count)), 4)

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(init_count) + ".png",
                                                       bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        a = dict()
        a[str(3287)] = matrix2
        sio.savemat("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(3287) + ".mat", a)
        print(3287)
        plt.imshow(matrix2, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(3287) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        # calculating from matrix2 to matrix 4
        step = (4018 - 3287) // 9
        init_count = 3287

        for x in range(11, 20):
            a = dict()
            init_count = init_count + step
            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                    coeffs = np.polyfit(xs, ys, degree)
                    ffit = np.poly1d(coeffs)
                    current_matrix[i][j] = ffit(init_count)

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                xs = [2618, 3287, 4018]
                ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                coeffs = np.polyfit(xs, ys, degree)
                ffit = np.poly1d(coeffs)
                current_matrix[i][j] = ffit(3652)

        a[str(3652)] = current_matrix
        sio.savemat("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(3652) + ".mat", a)
        print(3652)
        plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(3652) + ".png",
                                                       bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a["real"+str(3652)] = matrix3
        sio.savemat("../data/comparison/sequencedPolynomial/" + str(degree) + "/real" + str(3652) + ".mat", a)
        print("real3652")
        plt.imshow(matrix3, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedPolynomial/" + str(degree) + "/real" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a[str(4018)] = matrix4
        sio.savemat("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(4018) + ".mat", a)
        print(4018)
        plt.imshow(matrix4, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedPolynomial/" + str(degree) + "/" + str(4018) + ".png",
                                                       bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

    def calculate_cubic_spline_function(self):
        matrix1 = self.matrixreader.list_matrices[0]
        matrix2 = self.matrixreader.list_matrices[1]
        matrix3 = self.matrixreader.list_matrices[2]
        matrix4 = self.matrixreader.list_matrices[3]
        size = matrix1.shape

        # calculating from matrix1 to matrix 2

        degree = 2

        a = dict()
        a[str(2618)] = matrix1
        sio.savemat("../data/comparison/sequencedCubic/" + str(2618) + ".mat", a)
        print(2618)
        plt.imshow(matrix1, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedCubic/" + str(2618) + ".png",
                                                       bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        step = (3287 - 2618) // 9
        init_count = 2618

        # calculating from matrix1 to matrix 2

        for x in range(2, 10):
            a = dict()
            init_count = init_count + step

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                    cs = CubicSpline(xs, ys)
                    current_matrix[i][j] = round(float(cs(init_count)), 4)

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedCubic/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedCubic/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        a = dict()
        a[str(3287)] = matrix2
        sio.savemat("../data/comparison/sequencedCubic/" + str(3287) + ".mat", a)
        print(3287)
        plt.imshow(matrix2, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedCubic/" + str(3287) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        # calculating from matrix2 to matrix 4
        step = (4018 - 3287) // 9
        init_count = 3287

        for x in range(11, 20):
            a = dict()
            init_count = init_count + step

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                    cs = CubicSpline(xs, ys)
                    current_matrix[i][j] = round(float(cs(init_count)), 4)

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedCubic/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedCubic/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        # calculating matrix
        current_matrix = np.zeros([size[0], size[1]])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                xs = [2618, 3287, 4018]
                ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                cs = CubicSpline(xs, ys)
                current_matrix[i][j] = round(float(cs(3652)), 4)

        a[str(3652)] = current_matrix
        sio.savemat("../data/comparison/sequencedCubic/" + str(3652) + ".mat", a)
        print(3652)
        plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedCubic/" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a["real"+str(3652)] = matrix3
        sio.savemat("../data/comparison/sequencedCubic/real" + str(3652) + ".mat", a)
        print("real3652")
        plt.imshow(matrix3, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedCubic/real" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a[str(4018)] = matrix4
        sio.savemat("../data/comparison/sequencedCubic/" + str(4018) + ".mat", a)
        print(4018)
        plt.imshow(matrix4, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedCubic/" + str(4018) + ".png",
                                                       bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

    def calculate_piecewise_linear_function(self):

        matrix1 = self.matrixreader.list_matrices[0]
        matrix2 = self.matrixreader.list_matrices[1]
        matrix3 = self.matrixreader.list_matrices[2]
        matrix4 = self.matrixreader.list_matrices[3]
        size = matrix1.shape

        # calculating from matrix1 to matrix 2

        degree = 2

        a = dict()
        a[str(2618)] = matrix1
        sio.savemat("../data/comparison/sequencedPieceWise/" + str(2618) + ".mat", a)
        print(2618)
        plt.imshow(matrix1, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedPieceWise/" + str(2618) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        step = (3287 - 2618) // 9
        init_count = 2618

        # calculating from matrix1 to matrix 2

        for x in range(2, 10):
            a = dict()
            init_count = init_count + step

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                    current_matrix[i][j] = round(float(np.interp(init_count, xs, ys)), 4)

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedPieceWise/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedPieceWise/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        a = dict()
        a[str(3287)] = matrix2
        sio.savemat("../data/comparison/sequencedPieceWise/" + str(3287) + ".mat", a)
        print(3287)
        plt.imshow(matrix2, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedPieceWise/" + str(3287) + ".png",
                                                       bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        # calculating from matrix2 to matrix 4

        step = (4018 - 3287) // 9
        init_count = 3287

        for x in range(11, 20):
            a = dict()
            init_count = init_count + step

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                    current_matrix[i][j] = round(float(np.interp(init_count, xs, ys)), 4)

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedPieceWise/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedPieceWise/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        # calculating matrix
        current_matrix = np.zeros([size[0], size[1]])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                xs = [2618, 3287, 4018]
                ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                current_matrix[i][j] = round(float(np.interp(3652, xs, ys)), 4)

        a[str(3652)] = current_matrix
        sio.savemat("../data/comparison/sequencedPieceWise/" + str(3652) + ".mat", a)
        print(3652)
        plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedPieceWise/" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a["real"+str(3652)] = matrix3
        sio.savemat("../data/comparison/sequencedPieceWise/real" + str(3652) + ".mat", a)
        print("real3652")
        plt.imshow(matrix3, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedPieceWise/real" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a[str(4018)] = matrix4
        sio.savemat("../data/comparison/sequencedPieceWise/" + str(4018) + ".mat", a)
        print(4018)
        plt.imshow(matrix4, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedPieceWise/" + str(4018) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()


    def calculate_lagrange_interpolation(self):
        matrix1 = self.matrixreader.list_matrices[0]
        matrix2 = self.matrixreader.list_matrices[1]
        matrix3 = self.matrixreader.list_matrices[2]
        matrix4 = self.matrixreader.list_matrices[3]
        size = matrix1.shape

        # calculating from matrix1 to matrix 2

        degree = 2

        a = dict()
        a[str(2618)] = matrix1
        sio.savemat("../data/comparison/sequencedLagrange/" + str(2618) + ".mat", a)
        print(2618)
        plt.imshow(matrix1, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedLagrange/" + str(2618) + ".png",
                                                       bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        step = (3287 - 2618) // 9
        init_count = 2618

        # calculating from matrix1 to matrix 2

        for x in range(2, 10):
            a = dict()
            init_count = init_count + step

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                    lg = lagrange(xs, ys)
                    current_matrix[i][j] = round(float(lg(init_count)), 4)

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedLagrange/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedLagrange/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        a = dict()
        a[str(3287)] = matrix2
        sio.savemat("../data/comparison/sequencedLagrange/" + str(3287) + ".mat", a)
        print(3287)
        plt.imshow(matrix2, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedLagrange/" + str(3287) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        # calculating from matrix2 to matrix 4
        step = (4018 - 3287) // 9
        init_count = 3287

        for x in range(11, 20):
            a = dict()
            init_count = init_count + step

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                    lg = lagrange(xs, ys)
                    current_matrix[i][j] = round(float(lg(init_count)), 4)

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedLagrange/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedLagrange/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        # calculating matrix
        current_matrix = np.zeros([size[0], size[1]])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                xs = [2618, 3287, 4018]
                ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                lg = lagrange(xs, ys)
                current_matrix[i][j] = round(float(lg(3652)), 4)

        a[str(3652)] = current_matrix
        sio.savemat("../data/comparison/sequencedLagrange/" + str(3652) + ".mat", a)
        print(3652)
        plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedLagrange/" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a["real"+str(3652)] = matrix3
        sio.savemat("../data/comparison/sequencedLagrange/real" + str(3652) + ".mat", a)
        print("real3652")
        plt.imshow(matrix3, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedLagrange/real" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a[str(4018)] = matrix4
        sio.savemat("../data/comparison/sequencedLagrange/" + str(4018) + ".mat", a)
        print(4018)
        plt.imshow(matrix4, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedLagrange/" + str(4018) + ".png",
                                                       bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

    
    def calculate_nn_interpolation(self):
        matrix1 = self.matrixreader.list_matrices[0]
        matrix2 = self.matrixreader.list_matrices[1]
        matrix3 = self.matrixreader.list_matrices[2]
        matrix4 = self.matrixreader.list_matrices[3]
        size = matrix1.shape

        # calculating from matrix1 to matrix 2

        degree = 2

        a = dict()
        a[str(2618)] = matrix1
        sio.savemat("../data/comparison/sequencedNearestNeighbor/" + str(2618) + ".mat", a)
        print(2618)
        plt.imshow(matrix1, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedNearestNeighbor/" + str(2618) + ".png",
                                                       bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        step = (3287 - 2618) // 9
        init_count = 2618

        # calculating from matrix1 to matrix 2

        for x in range(2, 10):
            a = dict()
            init_count = init_count + step

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                    nn = NearestNDInterpolator((xs,xs), ys)
                    current_matrix[i][j] = round(float(nn(init_count,init_count)), 4)

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedNearestNeighbor/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedNearestNeighbor/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        a = dict()
        a[str(3287)] = matrix2
        sio.savemat("../data/comparison/sequencedNearestNeighbor/" + str(3287) + ".mat", a)
        print(3287)
        plt.imshow(matrix2, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedNearestNeighbor/" + str(3287) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        # calculating from matrix2 to matrix 4
        step = (4018 - 3287) // 9
        init_count = 3287

        for x in range(11, 20):
            a = dict()
            init_count = init_count + step

            # calculating matrix
            current_matrix = np.zeros([size[0], size[1]])

            for i in range(0, size[0]):
                for j in range(0, size[1]):
                    xs = [2618, 3287, 4018]
                    ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                    nn = NearestNDInterpolator((xs,xs), ys)
                    current_matrix[i][j] = round(float(nn(init_count,init_count)), 4)

            a[str(init_count)] = current_matrix
            sio.savemat("../data/comparison/sequencedNearestNeighbor/" + str(init_count) + ".mat", a)
            print(init_count)
            plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
            # plt.colorbar()
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            plt.savefig("../data/comparison/sequencedNearestNeighbor/" + str(init_count) + ".png",
                        bbox_inches='tight', pad_inches=0, transparent=True)
            #plt.show()

        # calculating matrix
        current_matrix = np.zeros([size[0], size[1]])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                xs = [2618, 3287, 4018]
                ys = [matrix1[i][j], matrix2[i][j], matrix4[i][j]]
                nn = NearestNDInterpolator((xs,xs), ys)
                current_matrix[i][j] = round(float(nn(3652,3652)), 4)

        a[str(3652)] = current_matrix
        sio.savemat("../data/comparison/sequencedNearestNeighbor/" + str(3652) + ".mat", a)
        print(3652)
        plt.imshow(current_matrix, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedNearestNeighbor/" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a["real"+str(3652)] = matrix3
        sio.savemat("../data/comparison/sequencedNearestNeighbor/real" + str(3652) + ".mat", a)
        print("real3652")
        plt.imshow(matrix3, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedNearestNeighbor/real" + str(3652) + ".png",
                    bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        a = dict()
        a[str(4018)] = matrix4
        sio.savemat("../data/comparison/sequencedNearestNeighbor/" + str(4018) + ".mat", a)
        print(4018)
        plt.imshow(matrix4, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.colorbar()
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.savefig("../data/comparison/sequencedNearestNeighbor/" + str(4018) + ".png",
                                                       bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()

        
    @staticmethod
    def get_mean_squared_error(real_path, simulated_path):

        mat_contents = sio.loadmat(real_path)
        base_name = os.path.basename(real_path)
        name_matrix = base_name.split('.mat')[0]
        name_content = name_matrix
        try:
            real_matrix = mat_contents[name_content]
        except:
            name_content = base_name.split('_T')[0]
            real_matrix = mat_contents[name_content]

        mat_contents = sio.loadmat(simulated_path)
        base_name = os.path.basename(simulated_path)
        name_matrix = base_name.split('.mat')[0]
        name_content = name_matrix
        try:
            simulated_matrix = mat_contents[name_content]
        except:
            name_content = base_name.split('_T')[0]
            simulated_matrix = mat_contents[name_content]

        diff = pow(real_matrix - simulated_matrix, 2)
        size = diff.shape
        diff = np.sum(diff)
        diff = diff/(size[0]*size[1])
        return diff



def main():
    path1 = "/home/aurea/Documents/Shell4DSimilarity/data/sequenced/DIP_seis_obs_234x326_T2618.mat"
    diff1 = 669
    path2 = "/home/aurea/Documents/Shell4DSimilarity/data/sequenced/DIP_seis_obs_234x326_T3287.mat"
    diff2 = 365
    path3 = "/home/aurea/Documents/Shell4DSimilarity/data/sequenced/DIP_seis_obs_234x326_T3652.mat"
    diff3 = 366
    path4 = "/home/aurea/Documents/Shell4DSimilarity/data/sequenced/DIP_seis_obs_234x326_T4018.mat"
    calculator = ComparisonDifferencesSyntheticCreator(path1, path2, path3, path4, diff1, diff2, diff3)

    calculator.calculate_linear_function()
    calculator.calculate_polynomial_function(1)
    calculator.calculate_polynomial_function(2)
    calculator.calculate_polynomial_function(3)
    calculator.calculate_polynomial_function(4)
    calculator.calculate_polynomial_function(5)
    calculator.calculate_polynomial_function(6)
    calculator.calculate_polynomial_function(7)
    calculator.calculate_polynomial_function(8)
    calculator.calculate_cubic_spline_function()
    calculator.calculate_piecewise_linear_function()
    calculator.calculate_lagrange_interpolation()
    calculator.calculate_nn_interpolation()

    print("Linear")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedLinear/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedLinear/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))

    print("Polynomial-1")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/1/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/1/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))

    print("Polynomial-2")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/2/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/2/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))

    print("Polynomial-3")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/3/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/3/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))

    print("Polynomial-4")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/4/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/4/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))

    print("Polynomial-5")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/5/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/5/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))

    print("Polynomial-6")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/6/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/6/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))

    print("Polynomial-7")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/7/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/7/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))

    print("Polynomial-8")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/8/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPolynomial/8/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))

    print("Cubic")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedCubic/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedCubic/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))

    print("PieceWise")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPieceWise/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedPieceWise/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))


    print("Lagrange")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedLagrange/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedLagrange/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))
    
    print("NearestNeighbor")
    real_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedNearestNeighbor/real3652.mat"
    simulated_path = "/home/aurea/Documents/Shell4DSimilarity/data/comparison/sequencedNearestNeighbor/3652.mat"
    print(calculator.get_mean_squared_error(real_path, simulated_path))


if __name__ == "__main__":
    main()






