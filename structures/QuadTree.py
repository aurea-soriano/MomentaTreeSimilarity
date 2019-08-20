import numpy as np
from features.OVMoments import OVMoments
from utils.SubMatrix import SubMatrix
from utils.MatrixReader import MatrixReader
from structures.Queue import Queue


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Node:
    def __init__(self, parent_, moments_, matrix_, depth_, weight_):
        self.parent = parent_
        self.moments = moments_
        self.matrix = matrix_
        self.topLeftChild = None
        self.topRightChild = None
        self.botLeftChild = None
        self.botRightChild = None
        self.depth = depth_
        self.weight = weight_
    
    def __iter__(self):
        return self



class QuadTree:
    def __init__(self, matrix_, max_depth):
        self.matrix = np.array(matrix_)
        self.total_depth = 0
        self.max_depth = max_depth
        moments = OVMoments(matrix_)
        print(moments.ovmoments)
        self.min_size = 4
        self.root = Node(None, moments.ovmoments, matrix_, 1, 1)
        self.large_moment = []
        if self.root.matrix.shape[0] <= self.min_size or self.root.matrix.shape[1] <= self.min_size:
            return
        else:
            self.root.depth = 1
            self.total_depth = 1
            self.insert_node(self.root)
            
    def insert_node(self, parent):
        if parent.matrix.shape[0] <= self.min_size or parent.matrix.shape[1] <= self.min_size:
            return

        if parent.depth > self.max_depth:
            return

        if int(self.total_depth) < int(parent.depth):
            self.total_depth = parent.depth
            
        parent_matrix = np.array(parent.matrix)
        # print(parent_matrix.shape)
        height = round(parent_matrix.shape[0] / 2)  # rows
        width = round(parent_matrix.shape[1] / 2)  # cols
        sub_matrix = SubMatrix()

        # calculating the sub matrices and their moments
        # top left
        top_left_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, 0, 0, width - 1, height - 1)).copy()
        top_left_moments = OVMoments(top_left_matrix)
        # print(top_left_matrix.shape)
        parent.topLeftChild = Node(parent, top_left_moments.ovmoments, top_left_matrix, parent.depth + 1, parent.weight/4)
        self.insert_node(parent.topLeftChild)

        # bottom left
        bot_left_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, 0, height, width-1, (height * 2) - 1)).copy()
        bot_left_moments = OVMoments(bot_left_matrix)
        # print(botLeftMatrix.shape)
        parent.botLeftChild = Node(parent, bot_left_moments.ovmoments, bot_left_matrix, parent.depth + 1, parent.weight/4)
        self.insert_node(parent.botLeftChild)

        # top right
        top_right_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, width, 0, (width * 2) - 1, height - 1))
        top_right_moments = OVMoments(top_right_matrix)
        # print(topRightMatrix.shape)
        parent.topRightChild = Node(parent, top_right_moments.ovmoments, top_right_matrix, parent.depth + 1, parent.weight/4)
        self.insert_node(parent.topRightChild)

        # bot right
        bot_right_matrix = np.array(sub_matrix.get_submatrix(parent_matrix, width, height, (width * 2) - 1,
                                                             (height * 2) - 1))
        bot_right_moments = OVMoments(bot_right_matrix)
        # print(botRightMatrix.shape)
        parent.botRightChild = Node(parent, bot_right_moments.ovmoments, bot_right_matrix, parent.depth + 1, parent.weight/4)
        self.insert_node(parent.botRightChild)


    #Breadth-first search
    def get_large_moment(self, max_level):
        #self.large_moment = np.concatenate(self.large_moment, self.root.moments)

        # a FIFO open_set
        open_set = Queue()
        
        # an empty set to maintain visited nodes
        closed_set = set()
  
        # a dictionary to maintain meta information (used for path formation)
        # key -> (parent state, action to reach child)
        meta = dict()

        # initialize
        root = self.root
        meta[root] = (None, None)
        open_set.enqueue(root)
        self.large_moment = []

        # For each node on the current level expand and process, if no children 
        # (leaf) then unwind
        while not open_set.empty():

            subtree_root = open_set.dequeue()
            if (subtree_root is not None) and (subtree_root.depth<=max_level):
                for moment in subtree_root.moments:
                    self.large_moment.append(moment)
                
                # For each child of the current tree process
                #for (child, action) in subtree_root.:
          
                # The node has already been processed, so skip over it
                if subtree_root.topLeftChild in closed_set:
                    continue
                  
                # The child is not enqueued to be processed, so enqueue this level of
                # children to be expanded
                if not open_set.contains(subtree_root.topLeftChild):
                    open_set.enqueue(subtree_root.topLeftChild)              # enqueue these nodes
                
                
                ##
                # The node has already been processed, so skip over it
                if subtree_root.topRightChild in closed_set:
                    continue
                  
                # The child is not enqueued to be processed, so enqueue this level of
                # children to be expanded
                if not open_set.contains(subtree_root.topRightChild):
                    open_set.enqueue(subtree_root.topRightChild)              # enqueue these nodes
                                   
                ##
                # The node has already been processed, so skip over it
                if subtree_root.botLeftChild in closed_set:
                    continue
                  
                # The child is not enqueued to be processed, so enqueue this level of
                # children to be expanded
                if not open_set.contains(subtree_root.botLeftChild):
                    open_set.enqueue(subtree_root.botLeftChild)              # enqueue these nodes
                    
                ##
                # The node has already been processed, so skip over it
                if subtree_root.botRightChild in closed_set:
                    continue
                  
                # The child is not enqueued to be processed, so enqueue this level of
                # children to be expanded
                if not open_set.contains(subtree_root.botRightChild):
                    open_set.enqueue(subtree_root.botRightChild)              # enqueue these nodes
                
                
                # We finished processing the root of this subtree, so add it to the closed 
                # set
                closed_set.add(subtree_root)
        return self.large_moment
                
def main():
    mat1_path = "/home/aurea/Documents/Shell4DSimilarity/data/size/1.mat"
    path_list = list()
    path_list.append(mat1_path)
    matrix_reader = MatrixReader(path_list)
    QuadTree(matrix_reader.list_matrices[0], 100)


if __name__ == "__main__":
    main()
