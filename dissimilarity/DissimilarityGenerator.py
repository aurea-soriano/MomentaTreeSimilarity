from dissimilarity.CityBlock import CityBlock
from dissimilarity.CosineBased import CosineBased
from dissimilarity.Euclidean import Euclidean
from dissimilarity.EuclideanSquared import EuclideanSquared
from dissimilarity.ExtendedJaccard import ExtendedJaccard
from dissimilarity.InfinityNorm import InfinityNorm
from dissimilarity.DynamicTimeWarping import DynamicTimeWarping
from dissimilarity.MaxMovingEuclidean import MaxMovingEuclidean
from dissimilarity.MinMovingEuclidean import MinMovingEuclidean
from dissimilarity.MeanSquaredError import MeanSquaredError
from dissimilarity.StandardMeanSquaredError import StandardMeanSquaredError


class DissimilarityGenerator:
    @staticmethod
    def get_dissimilarity_instance(dissimilarity_name):
        if dissimilarity_name == "City-block":
            return CityBlock()
        elif dissimilarity_name == "Cosine-based dissimilarity":
            return CosineBased()
        elif dissimilarity_name == "Euclidean":
            return Euclidean()
        elif dissimilarity_name == "Euclidean Squared":
            return EuclideanSquared()
        elif dissimilarity_name == "Extended Jaccard":
            return ExtendedJaccard()
        elif dissimilarity_name == "Infinity Norm":
            return InfinityNorm()
        elif dissimilarity_name == "Dynamic Time Warping (DTW)":
            return DynamicTimeWarping()
        elif dissimilarity_name == "Max Moving Euclidean":
            return MaxMovingEuclidean()
        elif dissimilarity_name == "Min Moving Euclidean":
            return MinMovingEuclidean()
        elif dissimilarity_name == "Mean Squared Error":
            return MeanSquaredError()
        elif dissimilarity_name == "Standard Mean Squared Error":
            return StandardMeanSquaredError()
        else:
            return Euclidean()
