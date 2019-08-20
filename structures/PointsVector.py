from math import sqrt
import numpy as np


class PointsVector:
    def __init__(self, vector, identifier, klass):
        # assert vector, "ERROR: vector can not be null"
        self.values = np.array(vector, dtype=float)
        self.size = len(vector)
        self.identifier = identifier
        self.klass = klass
        self.updateNorm = True
        self.norm = 0.0
        self.update_norm()
        self.DELTA = 0.00001

    def update_norm(self):
        self.norm = 0.0
        length = len(self.values)
        for i in range(0, length):
            self.norm += float(self.values[i]) * float(self.values[i])
        self.norm = sqrt(float(self.norm))
        self.updateNorm = False

    def dot(self, dense_vector2):
        return np.dot(self.values, dense_vector2.values)

    def calculate_norm(self):
        if self.updateNorm:
            self.update_norm()
        return self.norm

    def should_update_norm(self):
        self.updateNorm = True

    def is_null(self):
        for i in range(0, len(self.values)):
            if abs(float(self.values[i])) > 0.0:
                return False
        return True

    def get_value(self, index):
        assert (index <= self.size), "ERROR: index is greater than size"
        return float(self.values[index])

    def set_value(self, index, value):
        assert (index <= self.size),  "ERROR: index is greater than size"
        self.updateNorm = True
        self.values[index] = float(value)

    def normalize(self):
        assert (self.norm != 0.0), "ERROR: it is not possible to normalize a null vector"

        if self.calculate_norm() > self.DELTA:
            length = len(self.values)
            for i in range(0, length):
                self.values[i] = float(self.values[i]) / self.calculate_norm()
            self.norm = 1.0
        else:
            self.norm = 0.0
