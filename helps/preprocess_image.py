import cv2
import numpy as np

def preprocess_region(region):
    min_size = 50
    height, width = region.shape[:2]
    if height < min_size or width < min_size:
        scale = min_size / min(height, width)
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    alpha = 1.5
    beta = 20
    region = cv2.convertScaleAbs(region, alpha=alpha, beta=beta)
    
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    region = cv2.filter2D(region, -1, kernel)
    
    if len(region.shape) == 3:
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    return region