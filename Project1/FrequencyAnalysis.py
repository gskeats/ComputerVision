import SpatialFilter
import numpy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski



def converttogreyscale(image_array):
    image_array=cv2.cvtColor(image_array,cv2.COLOR_RGB2GRAY)
    return image_array
def checkgreyscale(image_array):
    if len(image_array.shape)>2:
        image_array=converttogreyscale(image_array)
    return image_array
def getdft(image_array):
    DFT2D = numpy.fft.fft2(image_array.astype(float))
    return DFT2D
def plot(coefficients,image_array):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Y = (numpy.linspace(-int(image_array.shape[0] / 2), int(image_array.shape[0] / 2) - 1, image_array.shape[0]))
    X = (numpy.linspace(-int(image_array.shape[1] / 2), int(image_array.shape[1] / 2) - 1, image_array.shape[1]))
    X, Y = numpy.meshgrid(X, Y)
    ax.plot_surface(X, Y, numpy.fft.fftshift(numpy.abs(coefficients)), cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot_surface(X, Y, numpy.fft.fftshift(numpy.log(numpy.abs(coefficients) + 1)), cmap=plt.cm.coolwarm, linewidth=0,
                    antialiased=False)
    plt.show()
    save(plt,"BasicFourier")

def plotlogmagnitude(coefficients,magplotname,logplotname):
    magnitudeImage = numpy.fft.fftshift(numpy.abs(coefficients))
    magnitudeImage = magnitudeImage / magnitudeImage.max()  # scale to [0, 1]
    magnitudeImage = ski.img_as_ubyte(magnitudeImage)
    cv2.imshow('Magnitude plot', magnitudeImage)
    logMagnitudeImage = numpy.fft.fftshift(numpy.log(numpy.abs(coefficients) + 1))
    logMagnitudeImage = logMagnitudeImage / logMagnitudeImage.max()  # scale to [0, 1]
    logMagnitudeImage = ski.img_as_ubyte(logMagnitudeImage)
    cv2.imshow('Log Magnitude plot', logMagnitudeImage)
    cv2.waitKey(0)
    SpatialFilter.saveimg(magplotname,magnitudeImage)
    SpatialFilter.saveimg(logplotname,logMagnitudeImage)

def save(plt,name):
    plt.savefig("/Users/grahamskeats/Programming_Projects/ComputerVision/Project1/"+name)


#first_image=SpatialFilter.loadimage("/Users/grahamskeats/Desktop/Noisy Puppy.JPG")
#SpatialFilter.displayimage(first_image,"original")
#greyscale=checkgreyscale(first_image)
#SpatialFilter.displayimage(greyscale,"greyscale")
#coefficients=getdft(greyscale)
#plot(coefficients,greyscale)
