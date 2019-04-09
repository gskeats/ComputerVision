import cv2
import numpy

def loadimage(image_name):
    pixel_array=cv2.imread(image_name)
    return pixel_array
def displayimage(pixel_array,window_number):
    cv2.imshow(window_number,pixel_array)
    cv2.waitKey(0)
def convertcolormodeltoHSV(pixel_array):
    cv2.cvtColor(pixel_array,cv2.COLOR_RGB2HSV)
    return pixel_array
def convertcolormodetoYUV(pixel_array):
    cv2.cvtColor(pixel_array,cv2.COLOR_RGB2YUV)
    return pixel_array
def getdimensions(pixel_array):
    return pixel_array.shape

first_image=loadimage("/Users/grahamskeats/Documents/Wake Forest/Junior Year/CSC 361/adirondack foliage.png")
HSV_image=convertcolormodeltoHSV(first_image)
YUV_image=convertcolormodetoYUV(first_image)
print(getdimensions(first_image))
