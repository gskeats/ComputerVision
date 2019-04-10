import image_splitter
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

def getkeypoints(harris_corner_frame):
    keypoints = numpy.argwhere(harris_corner_frame > 0.01 * harris_corner_frame.max())
    keypoints = [cv2.KeyPoint(point[1], point[0], 3) for point in keypoints]
    return keypoints


pixel_array=image_splitter.loadimage("./")