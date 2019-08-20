from numpy import *
import numpy as np


class Histogram:

    def __init__(self, original_matrix):
        self.original_matrix = original_matrix
        self.histogram = []
        self.calculate_histogram()

    def calculate_histogram(self):
        original_matrix = np.array(self.original_matrix)



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
