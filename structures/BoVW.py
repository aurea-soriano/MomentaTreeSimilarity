import numpy as np
#import cv2

class BoVW:

    def __init__(self, matrix_):
        self.matrix = np.array(matrix_)
        print("aurea")
        
        # defining feature extractor that we want to use KAZE, SIFT, ORB
        #sift = cv2.xfeatures2d.SIFT_create()
        #surf = cv2.xfeatures2d.SURF_create()
 
        #orb = cv2.ORB_create(nfeatures=1500)
        '''
 
        keypoints, descriptors = orb.detectAndCompute(img, None)
 
        img = cv2.drawKeypoints(img, keypoints, None)
 
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        
        
if __name__ == "__main__":
    matrix = [1, 2, 3, 1, 2, 3]
    bovw = BoVW(matrix)