from skimage.feature import hog
import image_splitter
from sklearn import svm

import cv2
def HOG(image):

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(4, 4), visualize=True, multichannel=True, block_norm= 'L2')
    return fd,hog_image

def make_labels(list_of_img):
    labels=[]
    for frame in list_of_img:
        image_splitter.displayimage(frame,"0")
        labels.append(input("label: "))


classifier=svm.SVC()

#mine = image_splitter.loadimage('./mine_train.JPG')

trees=image_splitter.loadimage('./trees.JPG')

pixel_array= image_splitter.loadimage('./agriculture.tif')

list_of_img=image_splitter.chop_img(pixel_array,50)
print(make_labels(list_of_img))
training_labels=['o','o','o','o','o','o','o','m','m',"m","o","o","o"]

x_train=[]
x_test=[]
for frame in range(len(training_labels)):
    features,image=HOG(list_of_img[frame])
    x_train.append(features)

classifier.fit(x_train,training_labels)


for frame in range(len(list_of_img)):
   features, image = HOG(list_of_img[frame])
   x_test.append(features)

prediction_matrix=classifier.predict(x_test)
print(prediction_matrix)
