import SpatialFilter
import FrequencyAnalysis
import FrequencyFiltering


#filter=SpatialFilter.definefilter(3)
#first_image=SpatialFilter.loadimage("/Users/grahamskeats/Desktop/Quiet Puppy.JPG")
#print(filter)
#new_image=SpatialFilter.apply_filter(filter, first_image)
#new_image=SpatialFilter.medianfilter(first_image,9)
#SpatialFilter.displayimage(first_image,"original")
#SpatialFilter.displayimage(new_image,"new")
#SpatialFilter.saveimg("3x3edgequietpuppyhoriz",new_image)
first_image=SpatialFilter.loadimage("/Users/grahamskeats/Desktop/Quiet Puppy.JPG")
SpatialFilter.displayimage(first_image,"original")
greyscale=FrequencyAnalysis.checkgreyscale(first_image)
SpatialFilter.displayimage(greyscale,"greyscale")
coefficients=FrequencyAnalysis.getdft(greyscale)
#FrequencyAnalysis.plot(coefficients,first_image)

#coefficients[0]=0
#coefficients[1]=0
#coefficients[2]=0

#coefficients[5:599]=0
#filtered_image=FrequencyFiltering.reconstructimage(coefficients)
#SpatialFilter.displayimage(filtered_image,"filtered")
#SpatialFilter.saveimg("almostallfreqto0",filtered_image)


#filteredimage=FrequencyFiltering.filterlowpass(coefficients,greyscale,cutoff=.05)
#filteredimage=FrequencyFiltering.butterworthfilter(coefficients,greyscale,highpass=1,cutoff=.75)
SpatialFilter.displayimage(filteredimage,"filtered")
coefficients=FrequencyAnalysis.getdft(filteredimage)
FrequencyAnalysis.plotlogmagnitude(coefficients,"highpassmagplot","highpasslogplot")
SpatialFilter.saveimg("highpasslarge",filteredimage)
#butterworthimagearray=butterworthfilter(coefficients,greyscale,highpass=1)
#SpatialFilter.displayimage(butterworthimagearray,"butterworth")
