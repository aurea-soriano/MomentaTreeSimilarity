from structures.PointsMatrix import PointsMatrix
from structures.DistanceMatrix import DistanceMatrix
from structures.MatlabMatrix import MatlabMatrix
from dissimilarity.DissimilarityGenerator import DissimilarityGenerator
from structures.QuadTree import QuadTree
from structures.AABSTree import AABSTree
from structures.PointsVector import PointsVector
from structures.NJTree import NJTree
from structures.BoVW import BoVW
import numpy as np
from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import math 
from timeit import default_timer as timer
from datetime import timedelta

class ProcessGenerator:

    def __init__(self, file_names, dissimilarity_name, dissimilarity_index,strategy_name,
                 strategy_index, radiobutton_option, localflag,
                 w1, h1, w2, h2, normalization_flag):
        self.file_names = file_names
        self.radiobutton_option = radiobutton_option
        self.dissimilarity_name = dissimilarity_name
        self.dissimilarity_index = dissimilarity_index
        self.strategy_name = strategy_name
        self.strategy_index = strategy_index
        self.dissimilarity_function = ""
        self.distmatrix = []
        self.pointsmatrix = []
        self.matmatrix = []
        self.njdismatrix = []
        # for tree similarity
        self.local_similarity = []
        self.feature_vectors = []
        self.normalization_flag = normalization_flag
        self.regression_std_error = 0.0

        if self.radiobutton_option == "pointsfile":
            self.pointsmatrix = PointsMatrix()
            self.pointsmatrix.load(self.file_names[0])
            self.dissimilarity_function = DissimilarityGenerator().get_dissimilarity_instance(self.dissimilarity_name)
            self.calculate_dismatrix_pointsmatrix()

        elif self.radiobutton_option == "dmatfile":
            self.distmatrix = DistanceMatrix()
            self.distmatrix.load(self.file_names[0])
            self.njdismatrix = self.distmatrix

        else:
            start = timer()
            # matlab files
            self.matmatrix = MatlabMatrix()
            self.matmatrix.load(self.file_names, normalization_flag)
            self.dissimilarity_function = DissimilarityGenerator().get_dissimilarity_instance(self.dissimilarity_name)
            if localflag is True:
                print("LocalFlag is True")
                self.apply_mask(w1, h1, w2, h2)

            if (self.strategy_name == "Point by Point"):
                print("Point by Point")
                self.calculate_dismatrix_matmatrix_point_by_point()
            elif (self.strategy_name == "Momenta Tree"):
                print("Momenta Tree")
                self.calculate_dismatrix_matmatrix() 
            elif (self.strategy_name == "Momenta AABB Tree"):
                self.calculate_dismatrix_matmatrix_aabs()
            elif (self.strategy_name == "Gradient"):
                self.calculate_distmatrix_gradients()
            end = timer()
            print("TIMEEEE")
            print(timedelta(seconds=end-start))
        
        '''         
        self.node_positions = []
        self.sources = []
        self.targets = []
        
        print("Distance calculated")
        start = timer()
        self.nj_tree = NJTree(self.njdismatrix)
        end = timer()
        print("TIMEEEE")
        print(timedelta(seconds=end-start))
            
        self.node_positions = self.nj_tree.node_positions
        self.sources = self.nj_tree.sources
        self.targets = self.nj_tree.targets
        print("NJ created")
        
       '''


    def calculate_number_vectors(self, tree_level):
        total_number = 0
        tree_level = tree_level - 1
        
        while tree_level>=0:
            total_number = total_number + pow(4, (tree_level))
            tree_level = tree_level - 1
        
        return total_number*7
        

    def calculate_regression(self):
        min_depth = 10000
        for quadTree in self.matmatrix.list_representations:
            if min_depth > int(quadTree.total_depth):
                min_depth = int(quadTree.total_depth)
        
        total_vectors = self.calculate_number_vectors(min_depth)
        self.feature_vectors = []
        
        for i in range(0, total_vectors):
            self.feature_vectors.append([])
        
        output   = []
        
        for index in range(0,len(self.matmatrix.list_representations)):
            quadTree = self.matmatrix.list_representations[index]                
            
            large_moment = quadTree.get_large_moment(min_depth)
            for i in range(0, len(large_moment)):
                self.feature_vectors[i].append(large_moment[i])
                
            label = str(self.matmatrix.list_labels[index])
            conc_number = ""
            for l in list(label):
                if l.isdigit():
                    conc_number = conc_number + str(l)
            label = int(conc_number)
            output.append(label)
        
        X = np.column_stack(self.feature_vectors)
        y = output
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)
        predictions1 = model.predict(X_test)
        
        # Perform 6-fold cross validation
        scores = cross_val_score(model, X, y, cv=2)
        
        print("Cross-validated scores:", scores)
        print("mean score:", sum(scores)/len(scores))
        
        predictions = cross_val_predict(model, X, y, cv=2)
        accuracy = metrics.r2_score(y, predictions)
        print("Cross-Predicted Accuracy:", accuracy)
        
        
        # standard error of the estimate
        # sqrt(sum(Y-Y')^2/N)
        error_sum = 0
        for i in range(0, len(predictions1)):
            error_sum = math.pow((y_test[i]-predictions1[i]),2)
        self.regression_std_error = math.sqrt(error_sum/len(predictions1))
        print("Standard error",self.regression_std_error )
        #plt.scatter(y_test,predictions)
        #plt.show()
        
        
        
    def apply_mask(self, w1, h1, w2, h2):
        mask = np.ones(self.matmatrix.list_matrices[0].shape)
        for i in range(0, self.matmatrix.list_matrices[0].shape[0]):
            for j in range(0, self.matmatrix.list_matrices[0].shape[1]):
                if h1 <= i <= h2 and w1 <= j <= w2:
                    mask[i][j] = 1
                else:
                    mask[i][j] = 0
        # plt.imshow(mask, cmap='nipy_spectral', vmin=0.0, vmax=1.0, interpolation='nearest')
        # plt.show()

        for i in range(0, len(self.matmatrix.list_matrices)):
            self.matmatrix.list_matrices[i] = self.matmatrix.list_matrices[i] * mask
            self.matmatrix.list_matrices[i] = self.matmatrix.list_matrices[i][w1:w2, h1:h2]
            # plt.imshow(self.matmatrix.list_matrices[i] , cmap='nipy_spectral', vmin=0.0, vmax=1.0,
            # interpolation='nearest')
            # plt.colorbar()
            # plt.axis('off')
            # plt.axes().get_xaxis().set_visible(False)
            # plt.axes().get_yaxis().set_visible(False)
            # plt.savefig("wells1/"+self.matmatrix.list_labels[i]+".png",
            #           bbox_inches='tight', pad_inches=0, transparent=True)
            # plt.show()

    def calculate_dismatrix_pointsmatrix(self):
        self.njdismatrix = DistanceMatrix()
        self.njdismatrix.labels = self.pointsmatrix.labels
        self.njdismatrix.ids = self.pointsmatrix.ids
        self.njdismatrix.classes = self.pointsmatrix.get_classes()
        self.njdismatrix.file_name = ""
        self.njdismatrix.nr_elements = len(self.pointsmatrix.instances)
        self.njdismatrix.max_distance = float("-inf")
        self.njdismatrix.min_distance = float("inf")
        self.njdismatrix.matrix = []

        for i in range(0, self.njdismatrix.nr_elements - 1):
            row = [0] * (i + 1)
            self.njdismatrix.matrix.append(row)

        for i in range(0, self.njdismatrix.nr_elements - 1):
            for j in range(0, i + 1):
                value = self.calculate_distance_trees(self.pointsmatrix.get_instance(i+1),
                                                                                    self.pointsmatrix.get_instance(j))
                value = round(float(value), 6)
                if (i + 1) < j:
                    self.njdismatrix.matrix[j - 1][i+1] = value
                else:
                    self.njdismatrix.matrix[i][j] = value

                if value > self.njdismatrix.max_distance and value >= 0.0:
                    self.njdismatrix.max_distance = value
                if 0.0 <= value < self.njdismatrix.min_distance:
                    self.njdismatrix.min_distance = value

    def calculate_dismatrix_matmatrix_point_by_point(self):
        self.njdismatrix = DistanceMatrix()
        self.njdismatrix.labels = self.matmatrix.list_labels
        self.njdismatrix.ids = []
        self.njdismatrix.classes = np.ones(len(self.matmatrix.list_matrices))
        self.njdismatrix.file_name = ""
        self.njdismatrix.nr_elements = len(self.matmatrix.list_matrices)
        self.njdismatrix.max_distance = float("-inf")
        self.njdismatrix.min_distance = float("inf")
        self.njdismatrix.matrix = []
        self.matmatrix.list_representations = []
        min_depth = 10000

        for i in range(0, self.njdismatrix.nr_elements -1):
            row = [0] * (i+1)
            self.njdismatrix.matrix.append(row)

        for i in range(0, self.njdismatrix.nr_elements-1):
            for j in range(0, i + 1):
                matrix1 = self.matmatrix.list_matrices[i+1]
                matrix2 = self.matmatrix.list_matrices[j]
                vector1 = np.ravel(matrix1)
                vector2 = np.ravel(matrix2)
                dense_vector1 = PointsVector(vector1, "", "")
                dense_vector2 = PointsVector(vector2, "", "")
                value = self.dissimilarity_function.calculate(dense_vector1, dense_vector2)
                value = round(float(value), 6)
                if (i + 1) < j:
                    self.njdismatrix.matrix[j - 1][i + 1] = value
                else:
                    self.njdismatrix.matrix[i][j] = value

                if value > self.njdismatrix.max_distance and value >= 0.0:
                    self.njdismatrix.max_distance = value
                if 0.0 <= value < self.njdismatrix.min_distance:
                    self.njdismatrix.min_distance = value

    def mean(self, matrix):
        sum = 0
        count = 1
        for i in range(0,len(matrix)):
            for j in range(0, len(matrix[i])):
                sum += matrix[i][j]
                count += 1
        return sum/count

    def std(self, matrix, mean):
        sum = 0
        count = 1
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                sum += pow(matrix[i][j] - mean, 2)
                count += 1
        return np.sqrt(sum/(count - 1))

    def zscore(self, matrix, mean, std):
        for i in range(0,len(matrix)):
            for j in range(0, len(matrix[i])):
                matrix[i][j] = (matrix[i][j] - mean)/std
        return matrix

    def calculate_dismatrix_matmatrix(self):
        self.njdismatrix = DistanceMatrix()
        self.njdismatrix.labels = self.matmatrix.list_labels
        self.njdismatrix.ids = []
        matrices_length = len(self.matmatrix.list_matrices)
        self.njdismatrix.classes = np.ones(matrices_length)
        self.njdismatrix.file_name = ""
        self.njdismatrix.nr_elements = matrices_length
        self.njdismatrix.max_distance = float("-inf")
        self.njdismatrix.min_distance = float("inf")
        self.njdismatrix.matrix = []
        self.matmatrix.list_representationses = []
        min_depth = 10000

        for i in range(0, self.njdismatrix.nr_elements):
            mat_matrix = np.array(np.nan_to_num(self.matmatrix.list_matrices[i]))
            quadTree = QuadTree(mat_matrix, 1000)
            self.matmatrix.list_representations.append(quadTree)
            if min_depth > int(quadTree.total_depth):
                min_depth = int(quadTree.total_depth)
            self.njdismatrix.ids.append((i+1))

        for i in range(0, self.njdismatrix.nr_elements -1):
            row = [0] * (i+1)
            self.njdismatrix.matrix.append(row)

        for i in range(0, self.njdismatrix.nr_elements-1):
            for j in range(0, i + 1):
                value = self.calculate_distance_trees(self.matmatrix.list_representations[i+1],
                                                              self.matmatrix.list_representations[j], min_depth)
                value = round(float(value), 6)
                if (i+1) < j:
                    self.njdismatrix.matrix[j - 1][i+1] = value
                else:
                    self.njdismatrix.matrix[i][j] = value

                if value > self.njdismatrix.max_distance and value >= 0.0:
                    self.njdismatrix.max_distance = value
                if 0.0 <= value < self.njdismatrix.min_distance:
                    self.njdismatrix.min_distance = value


    def calculate_distmatrix_gradients(self):
        self.njdismatrix = DistanceMatrix()
        self.njdismatrix.labels = self.matmatrix.list_labels
        self.njdismatrix.ids = []
        matrices_length = len(self.matmatrix.list_matrices)
        self.njdismatrix.classes = np.ones(matrices_length)
        self.njdismatrix.file_name = ""
        self.njdismatrix.nr_elements = matrices_length
        self.njdismatrix.max_distance = float("-inf")
        self.njdismatrix.min_distance = float("inf")
        self.njdismatrix.matrix = []
        self.matmatrix.list_representations = []
        min_depth = 10000

        for i in range(0, self.njdismatrix.nr_elements):
            mat_matrix = np.array(np.nan_to_num(self.matmatrix.list_matrices[i]))
            sizes = mat_matrix.shape
            result_gradient1 = np.zeros(sizes, np.float32)
            result_gradient1 = np.array(np.gradient(mat_matrix, axis=0))
            result_gradient2 = np.zeros(sizes, np.float32)
            result_gradient2 = np.array(np.gradient(mat_matrix, axis=1))
            print(result_gradient1.shape)
            print(result_gradient2.shape)
            plt.imshow(mat_matrix, cmap="jet")
            plt.show()
            plt.imshow(result_gradient1, cmap="jet")
            plt.show()
            plt.imshow(result_gradient2, cmap="jet")
            plt.show()
        
        
        
        #mean = self.mean(self.njdismatrix.matrix)
        #std = self.std(self.njdismatrix.matrix, mean)
        #self.njdismatrix.matrix = self.zscore(self.njdismatrix.matrix, mean, std)

    # "   A      B" \
    # "B  d(BA)   " \
    # "C  d(CA)  d(CB)"

    def calculate_dismatrix_matmatrix_aabs(self):
        self.njdismatrix = DistanceMatrix()
        self.njdismatrix.labels = self.matmatrix.list_labels
        self.njdismatrix.ids = []
        self.njdismatrix.classes = np.ones(len(self.matmatrix.list_matrices))
        self.njdismatrix.file_name = ""
        self.njdismatrix.nr_elements = len(self.matmatrix.list_matrices)
        self.njdismatrix.max_distance = float("-inf")
        self.njdismatrix.min_distance = float("inf")
        self.njdismatrix.matrix = []
        self.matmatrix.list_representations = []
        min_depth = 10000

        for i in range(0, self.njdismatrix.nr_elements):
            mat_matrix = np.array(np.nan_to_num(self.matmatrix.list_matrices[i]))
            aabstree = AABSTree(mat_matrix, 1000)
            self.matmatrix.list_representations.append(aabstree)
            if min_depth > int(aabstree.total_depth):
                min_depth = int(aabstree.total_depth)
            self.njdismatrix.ids.append((i+1))

        for i in range(0, self.njdismatrix.nr_elements -1):
            row = [0] * (i+1)
            self.njdismatrix.matrix.append(row)

        for i in range(0, self.njdismatrix.nr_elements-1):
            for j in range(0, i + 1):
                value = self.calculate_distance_trees_aabs(self.matmatrix.list_representations[i+1],
                                                              self.matmatrix.list_representations[j], min_depth)
                value = round(float(value), 6)
                if (i+1) < j:
                    self.njdismatrix.matrix[j - 1][i+1] = value
                else:
                    self.njdismatrix.matrix[i][j] = value

                if value > self.njdismatrix.max_distance and value >= 0.0:
                    self.njdismatrix.max_distance = value
                if 0.0 <= value < self.njdismatrix.min_distance:
                    self.njdismatrix.min_distance = value

    def calculate_distance_trees_aabs(self, tree1, tree2, max_depth):
        self.local_similarity = []
        self.traverse_aabstrees(tree1.root, tree2.root, max_depth)
        return np.sum(self.local_similarity) # sum

    def calculate_distance_trees(self, tree1, tree2, max_depth):
        self.local_similarity = []
        self.traverse_quadtrees(tree1.root, tree2.root, max_depth)
        return np.sum(self.local_similarity) # sum

    def traverse_quadtrees(self, node1, node2, max_depth):
        dense_vector1 = PointsVector(node1.moments, "", "")
        dense_vector2 = PointsVector(node2.moments, "", "")
        local_value = self.dissimilarity_function.calculate(dense_vector1, dense_vector2) # * node1.weight
        self.local_similarity.append(local_value)

        if node1.depth <= max_depth:
            if node1.topLeftChild is not None and node2.topLeftChild is not None:
                self.traverse_quadtrees(node1.topLeftChild, node2.topLeftChild, max_depth)

            if node1.topRightChild is not None and node2.topRightChild is not None:
                self.traverse_quadtrees(node1.topRightChild, node2.topRightChild, max_depth)

            if node1.botLeftChild is not None and node2.botLeftChild is not None:
                self.traverse_quadtrees(node1.botLeftChild, node2.botLeftChild, max_depth)

            if node1.botRightChild is not None and node2.botRightChild is not None:
                self.traverse_quadtrees(node1.botRightChild, node2.botRightChild, max_depth)

    def traverse_aabstrees(self, node1, node2, max_depth):
        dense_vector1 = PointsVector(node1.moments, "", "")
        dense_vector2 = PointsVector(node2.moments, "", "")
        local_value = 0 + self.dissimilarity_function.calculate(dense_vector1, dense_vector2)
        self.local_similarity.append(local_value)

        if node1.depth <= max_depth:
            if node1.child1 is not None and node2.child1 is not None:
                self.traverse_aabstrees(node1.child1, node2.child1, max_depth)

            if node1.child2 is not None and node2.child2 is not None:
                self.traverse_aabstrees(node1.child2, node2.child2, max_depth)
