from numpy import diff, sum, sqrt, fromfunction
import numpy as np


class OVMoments:

    def __init__(self, original_matrix):
        self.original_matrix = original_matrix
        self.ovmoments = []
        self.calculate_ovmoments()

    def calculate_ovmoments(self):
        original_matrix = np.array(self.original_matrix)
        area = sum(sum(original_matrix))
        dx = diff(original_matrix, 1, 1)
        dy = diff(original_matrix, 1, 0)
        mdx = sum(sum(abs(dx)))
        mdy = sum(sum(abs(dy)))
        x = original_matrix.shape[1] + 0.0
        y = original_matrix.shape[0] + 0.0

        # surface lengths
        product = original_matrix.shape[0] * original_matrix.shape[1]
        if product == 0:
            product = 0.00001
        lx = (1 + sum(1 + sum(sqrt(1 + (dx ** 2))))) / product
        ly = (1 + sum(1 + sum(sqrt(1 + (dy ** 2))))) / product

        # transversal stations
        px = (1.0 + sum(1.0 + sum(abs(dx) * fromfunction(lambda i, j: j, dx.shape) + 1.0))) / (0.1 + (mdx / 4.5))
        py = (1.0 + sum(1.0 + sum(abs(dy) * fromfunction(lambda i, j: i, dy.shape) + 1.0))) / (0.1 + (mdy / 8.0))

        # transversal Spatial Frequencies
        xx = x
        yy = y
        if xx == 0:
            xx = 0.00001
        if yy == 0:
            yy = 0.00001
        fx = (1 + sum(1 + sum(abs(dx), 1) / xx))
        fy = (1 + sum(1 + sum(abs(dy), 0) / yy))

        self.ovmoments.append(area)
        self.ovmoments.append(lx)
        self.ovmoments.append(ly)
        self.ovmoments.append(px)
        self.ovmoments.append(py)
        self.ovmoments.append(fx)
        self.ovmoments.append(fy)


def main():
    a = np.array([[-1, 0, 1], [1, 1, NaN]])
    value = abs(np.nanmin(a))+1
    print(value)
    print(a)
    a = [(x + value) for x in a]
    print(a)
    a = np.nan_to_num(a)
    print(a)
    a = [(x - 1) for x in a]
    print(a)
    ovmoment = OVMoments(a)
    print(ovmoment.ovmoments)


if __name__ == "__main__":
    main()
