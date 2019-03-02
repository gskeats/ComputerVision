import cv2
import numpy


def cornerHarris(frame):
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    frame=numpy.float32(frame)
    corners=getcorners(frame)
    map = corners > 0.01 * corners.max()
    frame[map] = 255
    return frame/frame.max()

def getcorners(frame):
    corners=cv2.cornerHarris(frame, blockSize=6, ksize=3, k=.04)
    corners = cv2.dilate(corners, None)
    return corners

def canny(frame,threshold1=600,threshold2=100):
    frame = cv2.Canny(frame,threshold1=threshold1,threshold2=threshold2)
    return frame

def get_video(function,key=0,name='test'):
    camera=cv2.VideoCapture(0)
    while True:
        ret,frame=camera.read()
        if function is not None:
            frame=function(frame)
        if key%256==32:
            break
        if key%256==114:
            return frame
        if key%256==115:
            file_name=input("filename: ")
            cv2.imwrite(file_name + ".jpg", frame)
        cv2.imshow(name,frame)
        key = cv2.waitKey(1)


def getkeypoints(harris_corner_frame):
    keypoints = numpy.argwhere(harris_corner_frame > 0.01 * harris_corner_frame.max())
    keypoints = [cv2.KeyPoint(point[1], point[0], 3) for point in keypoints]
    return keypoints