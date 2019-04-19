from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import cv2
import os
from skimage.feature import hog
import numpy
from time import time
from sklearn.neighbors import KNeighborsClassifier

def loadimage(image_name):
    pixel_array=cv2.imread(image_name)
    return pixel_array

def displayimage(pixel_array,window_number):
    cv2.imshow(window_number,pixel_array.astype(numpy.uint8))
    key=cv2.waitKey(0)
    if key % 256 == 115:
        file_name = input("filename: ")
        cv2.imwrite("./data_250/"+file_name + ".jpg",pixel_array)
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

def chop_img_set_size(img, dim):
    shape=getdimensions(img)
    list_arrays=[]
    for x_border in range(dim,shape[0],dim):
        for y_border in range(dim,shape[1],dim):
            list_arrays.append(img[x_border-dim:x_border,y_border-dim:y_border])
    return list_arrays


def label_and_write_images(file,pic_size):
    pixel_array= loadimage(file)
    list_images=chop_img_set_size(pixel_array,pic_size)
    for frame in list_images:
        displayimage(frame,"0")

def readdir(dirpath=None):
    if dirpath is None:
        dirpath=input("What is the path to the directory?")
    data_dir=os.listdir(dirpath)
    return data_dir

def trainsvm(train_data,train_labels,kernel='linear',slack=7):
    clf=svm.SVC(kernel=kernel,C=slack,probability=True)
    clf.fit(train_data,train_labels)
    return clf


def testsvm(clf,test_data,test_labels):
    accuracy=clf.score(test_data,test_labels)
    accuracy=accuracy*100
    print("The accuracy of the model was: "+str(accuracy)+"%")

def converttogreyscale(image_array):
    image_array=cv2.cvtColor(image_array,cv2.COLOR_RGB2GRAY)
    return image_array

def get_hog(image_path=None,image_array=None,channel=True):
    image=loadimage(image_path)
    if image_array is not None:
        image=image_array
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(4, 4), visualize=True, multichannel=False)
    return fd
def bulk_prediction(svm_hog,list_images):
    fd_list = []

    for image in list_images:
        fd_list.append(get_hog(image_array=image,channel=True))

    prediction_array = svm_hog.predict(fd_list)
    return prediction_array

def oneatatime(svm_hog,list_images):
    for image in list_images:
        fd = get_hog(image_array=image)
        prediction = svm_hog.predict([fd])
        if prediction == 1:
            print("got agri")
            displayimage(image, "1")

def oneatatimeprob(svm_hog,list_images):
    for image in list_images:
        fd = get_hog(image_array=image)
        prediction = svm_hog.predict_proba([fd])
        if abs(prediction[0][0]-prediction[0][1]) > .25 and prediction[0][0]<prediction[0][1]:
            print(prediction)
            displayimage(image, "1")




def runSVMHOG(train_data,train_labels,):


    svm_hog=trainsvm(train_data,train_labels)
    testsvm(svm_hog,test_data,test_labels)

    #oneatatimeprob(svm_hog,list_images)
    #print(bulk_prediction(svm_hog,list_images))

def ensemble_classification(training_data,training_labels,test_data,test_labels):
    forest = RandomForestClassifier(n_estimators=1000, max_depth=None,
                                 max_features=100, n_jobs=-1, random_state=0)
    forest.fit(training_data,training_labels)

    SVM=svm.SVC(kernel='linear',C=5,probability=True)
    SVM.fit(training_data,training_labels)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(training_data, training_labels)

    prediction=0
    prediction_array=[]

    for sample in test_data:
        prediction+=forest.predict([sample])[0]
        prediction+=SVM.predict([sample])[0]
        prediction+=neigh.predict([sample])[0]
        if prediction >=3:
            prediction=1
        else:
            prediction=0
        prediction_array.append(prediction)
    correct=0
    for prediction in range(0,len(prediction_array)):
        if prediction_array[prediction]==test_labels[prediction]:
            correct+=1
    print(prediction_array)
    print(test_labels)
    print("Accuracy "+str(correct/len(test_labels)))


def random_forest(training_data,training_labels,test_data,test_labels):
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                                 max_features=100, n_jobs=-1, random_state=0)
    t_start = time()
    clf.fit(train_data, train_labels)
    time_full_train = time() - t_start
    print(time_full_train)
    print(clf.score(test_data, test_labels))


labels=[]
data=[]
pix=loadimage("./agriculture.tif")
agr_list=readdir("./data_250/agriculture/")
other_list=readdir("./data_250/other/")

for image in other_list:
    pix_arr=loadimage("./data_250/other/"+image)
    pix_arr=converttogreyscale(pix_arr)
    features = get_hog(image_array=pix_arr)
    data.append(features)
    labels.append(0)

for image in agr_list:
    pix_arr=loadimage("./data_250/agriculture/"+image)
    pix_arr=converttogreyscale(pix_arr)
    features = get_hog(image_array=pix_arr)
    data.append(features)
    labels.append(1)


# test_size=.33
# random_list=[]
# split=(round(float(len(data))*test_size))


# for i in range(0,split):
#   random_list.append(random.randint(0,len(data)))

# for i in range(0,len(data)):
#    if i in random_list:
#        test_data.append(data[i])
#        test_labels.append(labels[i])
#    else:
#        train_data.append(data[i])
#        train_labels.append(labels[i])
combined=list(zip(data, labels))
numpy.random.shuffle(combined)
data,labels=zip(*combined)
test_data=data[-20:]
test_labels=labels[-20:]
train_data=data[:-20]
train_labels=labels[:-20]
ensemble_classification(train_data,train_labels,test_data,test_labels)

