import numpy as np
import cv2, imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
from math import atan2, cos, sin, sqrt, pi
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

main_threshold_value = 115
main_contour_threshold = 305000
main_blur = 100

def on_change_threshold(value):
    global main_threshold_value
    main_threshold_value = value

def on_change_contour_threshold(value):
    global main_contour_threshold
    main_contour_threshold = value

#setup realtime video frame
vid = cv2.VideoCapture(1, cv2.CAP_DSHOW)
ret, frame = vid.read()

cv2.imshow('main', frame)
cv2.createTrackbar('Threshold', 'main', 0, 255, on_change_threshold)
cv2.setTrackbarPos('Threshold', 'main', main_threshold_value)

cv2.createTrackbar('Contour Size', 'main', 0, 400000, on_change_contour_threshold)
cv2.setTrackbarPos('Contour Size', 'main', main_contour_threshold)

# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2688)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1520)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


#calibration vars
CHESS_BOARD_DIM = (9, 6)
SQUARE_SIZE = 14  # millimeters
OBJECT_POINTS = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), np.float32)
OBJECT_POINTS[:, :2] = np.mgrid[0 : CHESS_BOARD_DIM[0], 0 : CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
OBJECT_POINTS *= SQUARE_SIZE 

calibration_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Arrays to store object points and image points from all the images.
obj_points_3D = []  # 3d point in real world space
img_points_2D = []  # 2d points in image plane.

def calibrate_frame(image, chessboard_corners, image_grey=None):
    #if no grey image passed in, convert to greyscale
    if not image_grey:
         image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    chessboard_contours = cv2.cornerSubPix(image_grey, chessboard_corners, (3, 3), (-1, -1), calibration_criteria)
    obj_points_3D.append(OBJECT_POINTS)
    img_points_2D.append(chessboard_contours)

    image = cv2.drawChessboardCorners(image, CHESS_BOARD_DIM, chessboard_contours, True)
    return image



while 1:
    #read a frame from the video capture device
    ret, frame = vid.read()    

    #convert to grey scale
    main_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #check frame for calibration chessboard presence
    ret, calibration_corners = cv2.findChessboardCorners(main_grey, CHESS_BOARD_DIM)
    if ret:
        frame = calibrate_frame(frame, calibration_corners, image_grey=main_grey)
        

    #blur
    main_grey = cv2.GaussianBlur(main_grey, (7, 7), 0)

    #main_grey = cv2.Canny(main_grey, main_threshold_value, main_threshold_value+25)
    main_grey = cv2.dilate(main_grey, None, iterations=1)
    main_grey = cv2.erode(main_grey, None, iterations=1)

    # Convert the grayscale image to binary
    ret, main_binary = cv2.threshold(main_grey, main_threshold_value, 255, cv2.THRESH_BINARY)
    main_inverted_binary = ~main_binary

    contours, hierarchy = cv2.findContours(main_inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
    
    with_contours = frame.copy()
    
    #masking array
    mask = np.zeros(frame.shape, dtype="uint8")
    
    large_contours = []
    for i,cnt in enumerate(contours):
        #remove small contours
        if  cv2.contourArea(cnt) > main_contour_threshold:
            large_contours.append(cnt)
            
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(with_contours, [box.astype("int")], -1, (0, 255, 0), 2)
            
            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(with_contours, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            cv2.circle(with_contours, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(with_contours, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(with_contours, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(with_contours, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            cv2.line(with_contours, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(with_contours, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # draw the object sizes on the image
            cv2.putText(with_contours, f"{round(dA, 2)}px", (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(with_contours, f"{round(dB, 2)}px", (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            sz = len(cnt)
            data_pts = np.empty((sz, 2), dtype=np.float64)
            for i in range(data_pts.shape[0]):
                data_pts[i,0] = cnt[i,0,0]
                data_pts[i,1] = cnt[i,0,1]
            mean = np.empty((0))
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
            cntr = (int(mean[0,0]), int(mean[0,1]))
            cv2.circle(with_contours, cntr, 3, (255, 0, 255), 2)
            angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
            label = f"Angle: {round(np.rad2deg(angle) - 90,2)} deg"
            cv2.putText(with_contours, label, (cntr[0]+10, cntr[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            #masking and cropping out pouches
            
            #draw the contour in the mask
            cv2.fillPoly(mask, [box.astype("int")], (255,255,255))
            # mask = mask.astype("uint8")
            mask = cv2.bitwise_and(frame, mask)

            
    cv2.imshow("mask", cv2.resize(mask, (640, 360), interpolation= cv2.INTER_LINEAR))
    
    cv2.drawContours(with_contours, large_contours, -1,(255,0,255),1)

    cv2.imshow('main', cv2.resize(with_contours, (1280, 720), interpolation= cv2.INTER_LINEAR))

    #escape key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()