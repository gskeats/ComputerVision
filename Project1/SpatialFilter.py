import cv2
import numpy
from scipy.misc import imsave
def loadimage(image_name):
    pixel_array=cv2.imread(image_name)
    return pixel_array
def displayimage(pixel_array,window_number):
    cv2.imshow(window_number,pixel_array.astype(numpy.uint8))
    cv2.waitKey(0)
def convertcolormodeltoHSV(pixel_array):
    cv2.cvtColor(pixel_array,cv2.COLOR_RGB2HSV)
    return pixel_array
def convertcolormodetoYUV(pixel_array):
    cv2.cvtColor(pixel_array,cv2.COLOR_RGB2YUV)
    return pixel_array
def getdimensions(pixel_array):
    return pixel_array.shape
def definefilter(filter_size=3):
    filter=[]
    for pixel_space in range(filter_size):
        for current_pixel in range(filter_size):
            list=[1]*filter_size
            filter_value=int(input("Pixel number "+str(current_pixel)+" in list "+str(pixel_space)+" "))
            list[current_pixel]=filter_value
        filter.append(list)
    filter=numpy.array(filter)
    return filter
def definefilterusingones(filter_size):
    filter=numpy.ones((filter_size,filter_size))
    filter=filter/filter.sum()
    return filter

def apply_filter(defined_list_filter,image_array):
    filtered_image_array=numpy.ones(getdimensions(image_array))
    filtered_image_array[:,:,0]=cv2.filter2D(image_array[:,:,0],-1, defined_list_filter)
    filtered_image_array[:,:,1]=cv2.filter2D(image_array[:,:,1],-1, defined_list_filter)
    filtered_image_array[:,:,2]=cv2.filter2D(image_array[:,:,2],-1, defined_list_filter)
    filtered_image_array=scale_values(filtered_image_array)
    return filtered_image_array

def scale_values(image_array):
    image_array=image_array/image_array.max()
    image_array=image_array*255
    return image_array
def saveimg(filename,image_array,file_type="JPG"):
    imsave(filename+"."+file_type,image_array)

#filter=definefilter(3)
#first_image=loadimage("/Users/grahamskeats/Desktop/Noisy Puppy.JPG")
#print(filter)
#new_image=apply_filter(filter, first_image)
#displayimage(first_image,"original")
#displayimage(new_image,"new")
#saveimg("new_puppy",new_image)