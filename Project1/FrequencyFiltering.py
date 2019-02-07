import FrequencyAnalysis
import SpatialFilter
import numpy

def get_bowlD(image_array):
    Y = (numpy.linspace(-int(image_array.shape[0] / 2), int(image_array.shape[0] / 2) - 1, image_array.shape[0]))
    X = (numpy.linspace(-int(image_array.shape[1] / 2), int(image_array.shape[1] / 2) - 1, image_array.shape[1]))
    X, Y = numpy.meshgrid(X, Y)
    D = numpy.sqrt(X * X + Y * Y)
    return D

def filterlowpass(coefficients,image_array,cutoff=0.25):
    D = get_bowlD(image_array)
    D0 = cutoff * D.max()
    idealLowPass = D <= D0
    DFTFiltered = coefficients * numpy.fft.fftshift(idealLowPass)
    image_array = numpy.abs(numpy.fft.ifft2(DFTFiltered))
    return image_array

def butterworthfilter(coefficients,image_array,highpass=0,cutoff=0.25):
    for n in range(1, 5):
        # Create Butterworth filter of order n
        D=get_bowlD(image_array)
        D0 = cutoff * D.max()

        H = 1.0 / (1 + (numpy.sqrt(2) - 1) * numpy.power(D / D0, 2 * n))
        H=H+highpass
        butterworthfiltered= coefficients * numpy.fft.fftshift(H)
        butterworthfilterecon = numpy.abs(numpy.fft.ifft2(butterworthfiltered))
        return butterworthfilterecon

def reconstructimage(coefficients):
    image_array=numpy.abs(numpy.fft.ifft2(coefficients))
    return image_array

#first_image=SpatialFilter.loadimage("/Users/grahamskeats/Desktop/Noisy Puppy.JPG")
#greyscale=FrequencyAnalysis.checkgreyscale(first_image)
#SpatialFilter.displayimage(greyscale,"grey")
#coefficients=FrequencyAnalysis.getdft(greyscale)
#filteredimage=filterlowpass(coefficients,greyscale)
#SpatialFilter.displayimage(filteredimage,"filtered")
#coefficients=FrequencyAnalysis.getdft(filteredimage)
#FrequencyAnalysis.plotlogmagnitude(coefficients,"ideallowpassmagplot","ideallowpasslogplot")
#SpatialFilter.saveimg("ideallowpassimage",filteredimage)
#butterworthimagearray=butterworthfilter(coefficients,greyscale,highpass=1)
#SpatialFilter.displayimage(butterworthimagearray,"butterworth")