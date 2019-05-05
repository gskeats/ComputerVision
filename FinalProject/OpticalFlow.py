import cv2
import numpy as np
traverse_point = []


def canny(frame,threshold1=600,threshold2=100):
    frame = cv2.Canny(frame,threshold1=threshold1,threshold2=threshold2)
    return frame

def get_video(function,key=0,name='test'):
    capture=False
    camera=cv2.VideoCapture(0)
    while True:
        ret,frame=camera.read()
        if function is not None and capture is False:
            frame=function(frame)
        if key%256==32:
            break
        if key%256==114 and capture is False:
            hist=get_hist(frame)
            capture=True
        if capture is True:
            frame=track(frame,hist)
        if key%256==115:
            file_name=input("filename: ")
            cv2.imwrite(file_name + ".jpg", frame)
        cv2.imshow(name,frame)
        key = cv2.waitKey(1)

def draw_rect(frame):
    rows, cols, _ = frame.shape
    global hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(9):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame
def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(9):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 200, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)
def get_hist(frame):
    hist=hand_histogram(frame)
    return hist
def track(frame,hist):
    frame=manage_image_opr(frame,hist)
    return frame
def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None
def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)
def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None
def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def max_contour(contour_list):
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

        return contour_list[max_i]
def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_circles(frame, traverse_point)

    return frame

def converttogreyscale(image_array):
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return image_array

def optical_flow_dense(function=None,key=0):
    capture=False
    camera=cv2.VideoCapture(0)
    ret,previous=camera.read()
    previous = converttogreyscale(previous)
    while True:
        ret,frame=camera.read()
        if function is not None:
            frame=function(frame)
        if key%256==32:
            break

        if key%256==115:
            file_name=input("filename: ")
            cv2.imwrite(file_name + ".jpg", frame)
            cv2.imwrite(file_name + "motion.jpg", bgr)
        curr=converttogreyscale(frame)
        flow=cv2.calcOpticalFlowFarneback(curr,previous,None,.5,30,15,10,5,1.2,0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        previous=curr
        cv2.imshow("live",frame)
        cv2.imshow("optical_flow",bgr)
        key = cv2.waitKey(1)
def optical_flow_sparse(function=None,key=0):
    capture=False
    color = np.random.randint(0, 255, (100, 3))

    camera=cv2.VideoCapture(0)
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ret, old_frame = camera.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)

    while True:
        ret,frame=camera.read()
        if function is not None:
            frame=function(frame)
        if key%256==32:
            break

        if key%256==115:
            file_name=input("filename: ")
            cv2.imwrite(file_name + ".jpg", frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        cv2.imshow("live",img)
        key = cv2.waitKey(5)
