import numpy as np


class MinMaxNormalization:

    @staticmethod
    def normalize(matrix, old_min, old_max, new_min, new_max):
        div = old_max - old_min
        if div==0:
            div = 0.0001
        norm_matrix = [(((x - old_min) * (new_max - new_min)) / div) + new_min for x in matrix]
        return norm_matrix
    
    @staticmethod
    def normalize_value(value, old_min, old_max, new_min, new_max):
        div = old_max - old_min
        if div==0:
            div = 0.0001
        norm_value = (((value - old_min) * (new_max - new_min)) / div) + new_min
        return norm_value

def main():
    matrix = [[0.0, 400.0], [np.NaN, 100.0], [-30.0, np.NaN]]
    max_value = np.nanmax(matrix)
    min_value = np.nanmin(matrix)
    norm_matrix = MinMaxNormalization.normalize(matrix, min_value, max_value, 12, 13)
    print(norm_matrix)
    
    original_matrix = np.array(matrix) 
    original_matrix = original_matrix + 31
    print(original_matrix)
    original_matrix = np.nan_to_num(original_matrix)
    print(original_matrix)
    original_matrix = np.array(original_matrix, dtype=np.float32)
    print(original_matrix)


if __name__ == "__main__":
    main()
