from math import sqrt
from structures.PointsVector import PointsVector


class DynamicTimeWarping:

    def __init__(self):
        self.warpingWindowPerc = 0.25
        self.g = [[]]

    def dist(self, value1, value2):
        diff = value1 - value2
        return sqrt(diff * diff)

    def calculate(self,  dense_vector1, dense_vector2):
        vector1 = dense_vector1.values
        vector2 = dense_vector2.values
        assert len(vector1) == len(vector2), "vectors must be of equal sizes"

        # creating the matrix
        self.g = []
        for i in range(0, len(vector2)):
            row = []
            for j in range(0, len(vector1)):
                row.append(0.0)
            self.g.append(row)

        # initial condition
        self.g[0][0] = self.dist(vector1[0], vector2[0])

        r = int((len(vector1) - 1) * self.warpingWindowPerc)

        # filling the diagonals
        j = 0
        for i in range(r+1, len(vector2)):
            self.g[j][i] = float('inf')
            self.g[i][j] = float('inf')
            j += 1

        # calculate the first row and column
        for i in range(1, r+1):
            self.g[0][i] = self.g[0][i - 1] + self.dist(vector1[i], vector2[0])
            self.g[i][0] = self.g[i - 1][0] + self.dist(vector1[0], vector2[i])

        # calculate the remaining values
        for i in range(1, len(vector2)):
            min_value = max(1, i - r)
            max_value = min(len(vector1) - 1, i+r)
            for j in range(min_value, max_value + 1):
                self.g[i][j] = min(min(self.g[i - 1][j], self.g[i - 1][j - 1]), self.g[i][j - 1]) + \
                               self.dist(vector1[j], vector2[i])

        return self.g[len(vector1) - 1][len(vector2) - 1]


def main():
    abstract_dissimilarity = DynamicTimeWarping()
    vector1 = [7876206.000000, 2.721705, 2.482324, 510.317183, 1310.820459, 907.367521, 594.858896]
    vector2 = [7537091.000000, 3.262750, 3.004038, 518.909965, 1310.309662, 1107.230769, 733.423313]
    vector3 = [7421256.000000, 3.462628, 3.211842, 524.057260, 1306.746288, 1179.025641, 786.895706]
    vector4 = [7284190.000000, 3.630403, 3.402637, 523.408172, 1303.025129, 1241.683761, 837.233129]
    dense_vector1 = PointsVector(vector1, 1, "")
    dense_vector2 = PointsVector(vector2, 2, "")
    dense_vector3 = PointsVector(vector3, 3, "")
    dense_vector4 = PointsVector(vector4, 4, "")
    print(abstract_dissimilarity.calculate(dense_vector1, dense_vector2))
    print(abstract_dissimilarity.calculate(dense_vector1, dense_vector3))
    print(abstract_dissimilarity.calculate(dense_vector1, dense_vector4))
    print(abstract_dissimilarity.calculate(dense_vector2, dense_vector3))
    print(abstract_dissimilarity.calculate(dense_vector2, dense_vector4))
    print(abstract_dissimilarity.calculate(dense_vector3, dense_vector4))


if __name__ == "__main__":
    main()
