import numpy
import cv2


def loadimage(image_name):
    pixel_array=cv2.imread(image_name)
    return pixel_array

def displayimage(pixel_array,window_number):
    cv2.imshow(window_number,pixel_array.astype(numpy.uint8))
    cv2.waitKey(0)
    cv2.destroyWindow(window_number)

def getdimensions(pixel_array):
    return pixel_array.shape

def saveimg(img,name=None):
    if name is None:
        file_name = input("filename: ")
    else:
        file_name=name
    cv2.imwrite(file_name + ".jpg", img)

def chop_img(img, num_regions):
    shape=getdimensions(img)
    x_chunk_size=round(shape[0] / num_regions)
    y_chunk_size=round((shape[1] / num_regions))
    list_arrays=[]
    for x_border in range(x_chunk_size,shape[0],x_chunk_size):
        for y_border in range(y_chunk_size,shape[1],y_chunk_size):
            list_arrays.append(img[x_border-x_chunk_size:x_border,y_border-y_chunk_size:y_border])
    return list_arrays






