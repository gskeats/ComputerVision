import EdgesandCorners
import cv2

def write_kp(frame):
    kp,des=getSIFT(frame)
    kp_frame = cv2.drawKeypoints(frame, kp[:],None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp_frame

def getSIFT(frame):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(frame,None)
    return kp,des

