from structures.PointsVector import PointsVector


class MeanSquaredError:

    def __init__(self):
        self.window_size = 3

    def calculate(self,  dense_vector1, dense_vector2):
        vector1 = dense_vector1.values
        vector2 = dense_vector2.values
        assert len(vector1) == len(vector2), "vectors must be of equal sizes"
        distance = ((vector1 - vector2) ** 2).mean()
        return distance


def main():
    abstract_dissimilarity = MeanSquaredError()
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
