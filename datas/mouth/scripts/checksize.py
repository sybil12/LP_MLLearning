import os
import sys
import cv2
for image in os.listdir(sys.argv[1]):
    img = cv2.imread(os.path.join(sys.argv[1],image))
    #print image," shape is ",img.shape
    #print(type(img.shape))
    if img.shape[0]<48 or img.shape[1]<48:
    	print(image," shape is ",img.shape)

