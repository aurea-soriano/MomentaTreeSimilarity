import numpy as np
import matplotlib.pyplot as plt


class SubMatrix:

    # return a sub matrix from x1,y1 to x2,y2 (including x2,y2)
    @staticmethod
    def get_submatrix(matrix, col1, row1, col2, row2):
        
        col1 = int(col1)
        col2 = int(col2)
        row1 = int(row1)
        row2 = int(row2)
        assert col1 >= 0 and row1 >= 0, "col1, row1 must be positive"
        assert col1 <= col2 and row1 <= row2, "col1,row1 must be smaller than col2,row2"

        m_shape = matrix.shape
        max_col2 = col2 if col2 < m_shape[1] else m_shape[1]-1
        max_row2 = row2 if row2 < m_shape[0] else m_shape[0]-1

        return matrix[row1:(max_row2+1)][col1:(max_col2+1)]
    # [[matrix[row][col] for col in range(col1, max_col2+1)]
    #           for row in range(row1, max_row2+1)]


def main():
    #a = np.array([[1, 1, 1], [1, 1, 1], [6, 1, 3], [8, 4, 10]])
    #utils = SubMatrix()
    #print(a)
    #print(a[3, 0])
    #print(utils.get_submatrix(a, 2, 2, 2, 8))
    print("test")
    tree_level = 6
    total_number = 0
    tree_level = tree_level - 1
       
    while tree_level>=0:
        total_number = total_number + pow(4, (tree_level))
        tree_level = tree_level - 1
        
    print(total_number*7)
    
    print(total_number)

if __name__ == "__main__":
    main()
