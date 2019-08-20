from structures.PointsVector import PointsVector


class PointsMatrix:

    def __init__(self):
        self.file_name = ""
        self.attributes = []
        self.labels = []
        self.ids = []
        self.instances = []
        self.class_flag = False

    def contains(self, dense_vector):
        if dense_vector in self.instances:
            return True
        else:
            return False

    def add_instance(self, dense_vector):
        self.instances.append(dense_vector)

    def set_instance(self, index, dense_vector):
        assert (len(self.instances) > index), "ERROR: wrong index or vector of wrong size"
        self.instances[index] = dense_vector

    def remove_instance(self, index):
        assert (len(self.instances) > index), "ERROR: wrong index"
        return self.instances.pop(index)

    def get_instance(self, index):
        assert (len(self.instances) > index), "ERROR: this row does not exists in the matrix"
        return self.instances[index]

    def normalize(self):
        size = len(self.instances)
        for i in range(0, size):
            if not self.instances[i].isNull():
                self.instances[i].normalize()

    def to_matrix(self):
        matrix = [[]]
        size = len(self.instances)
        for i in range(0, size):
            matrix.append(self.instances[i].values)
        return matrix

    def set_attributes(self, attributes):
        self.attributes = attributes

    def get_ids(self):
        ids = []
        for i in range(0, len(self.instances)):
            ids.append(self.instances[i].id)

    def get_classes(self):
        cdata = []
        for i in range(0, len(self.instances)):
            cdata.append(self.instances[i].klass)

    def save(self, file_name):
        file = open(file_name, "w+")
        file.write("D")
        if self.class_flag:
            file.write("Y\r\n")
        else:
            file.write("N\r\n")

        file.write(str(len(self.instances)))
        file.write("\r\n")
        file.write(str(len(self.attributes)))
        file.write("\r\n")

        # Writting the attributes
        if len(self.attributes) > 0:
            for i in range(0, len(self.attributes)):
                file.write((self.attributes[i].replace("<>", " ")).strip())
                if i < len(self.attributes) - 1:
                    file.write(";")
            file.write("\r\n")
        else:
            file.write("\r\n")

        # writting the vectors
        for i in range(0, len(self.instances)):
            dense_vector = self.instances[i]
            file.write(str(dense_vector.id))
            file.write(";")
            for j in range(0, len(dense_vector.values)):
                file.write(str(dense_vector.get_value(i)))
                file.write(";")
            file.write(str(dense_vector.klass))
            file.write("\r\n")
        file.flush()
        file.close()

    def load(self, file_name):
        self.file_name = file_name
        file = open(file_name, "r")
        count = 0
        index = 1
        self.attributes = []
        self.labels = []
        self.ids = []
        self.instances = []
        for line in file:
            count += 1
            if count == 1:
                option = line[-1:]
                if option == 'Y' or option == 'y':
                    self.class_flag = True
                else:
                    self.class_flag = False

            elif count == 2:
                number_instances = line

            elif count == 3:
                number_attributes = line

            elif count == 4:
                self.attributes = [x.strip() for x in line.split(';')]

            else:
                line_array = [x.strip() for x in line.split(';')]
                self.labels.append(line_array[0])
                self.ids.append(index)

                if self.class_flag:
                    dense_vector = PointsVector(line_array[1:(len(line_array)-1)], line_array[0],
                                                line_array[-1])
                    self.instances.append(dense_vector)

                else:
                    dense_vector = PointsVector(line_array[1:], line_array[0], 1.0)
                    self.instances.append(dense_vector)
                index += 1
