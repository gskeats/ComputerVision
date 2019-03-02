import cv2
import EdgesandCorners
import time
import SIFTDescrip


def matchkp(des1,des2):
    bfmatcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)
    matches=bfmatcher.match(des1,des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def match_keypoints_sift():
    frame1=EdgesandCorners.get_video(None,key=114)
    time.sleep(5)
    frame2=EdgesandCorners.get_video(None,key=114)
    keypointsframe1,descriptorframe1=SIFTDescrip.getSIFT(frame1)
    keypointsframe2,descriptorframe2=SIFTDescrip.getSIFT(frame2)
    matches=matchkp(descriptorframe1,descriptorframe2)
    matched_frame = cv2.drawMatches(frame1, keypointsframe1, frame2, keypointsframe2, matches[:10], None,flags=2)
    cv2.imshow("matched",matched_frame)
    key = cv2.waitKey(0)
    cv2.imwrite("keypointmatchpaintingsift.jpg",matched_frame)


def harrisCornerMatch():
    sift = cv2.xfeatures2d.SIFT_create()
    frame1=EdgesandCorners.get_video(None,key=114)
    frame1=cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)

    time.sleep(5)
    frame2=EdgesandCorners.get_video(None,key=114)
    frame2=cv2.cvtColor(frame2,cv2.COLOR_RGB2GRAY)


    cornerpoints1=EdgesandCorners.getcorners(frame1)
    cornerpoints2=EdgesandCorners.getcorners(frame2)


    keypoints1=EdgesandCorners.getkeypoints(cornerpoints1)
    keypoints2=EdgesandCorners.getkeypoints(cornerpoints2)


    keypoints1,descriptions1=sift.compute(frame1,keypoints1,None)
    keypoints2,descriptions2=sift.compute(frame2,keypoints2,None)


    matches=matchkp(descriptions1,descriptions2)
    matched_frame = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches[:10], None,flags=2)
    cv2.imshow("matched",matched_frame)
    key = cv2.waitKey(0)
    cv2.imwrite("keypointmatchpaintingharris.jpg",matched_frame)

