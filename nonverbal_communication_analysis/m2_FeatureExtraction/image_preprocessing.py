from nonverbal_communication_analysis.environment import DATASET_SYNC
import numpy as np
import cv2

group_video_path = DATASET_SYNC / \
    '3CLC9VWRSAMPLE/task_1/Videopc305122018112450_sample.avi'

cap = cv2.VideoCapture(str(group_video_path))
fgbg1 = cv2.createBackgroundSubtractorMOG2()
fgbg2 = cv2.createBackgroundSubtractorKNN()

# while(1):
#     ret, frame = cap.read()

#     fgmask1 = fgbg1.apply(frame)
#     fgmask2 = fgbg2.apply(frame)

#     cv2.imshow('original', frame)
#     cv2.imshow('MOG2', fgmask1)
#     cv2.imshow('KNN', fgmask2)

#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# while(1):

#     # Take each frame
#     _, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     lower_red = np.array([30,150,50])
#     upper_red = np.array([255,255,180])

#     mask = cv2.inRange(hsv, lower_red, upper_red)
#     res = cv2.bitwise_and(frame,frame, mask= mask)

#     laplacian = cv2.Laplacian(frame,cv2.CV_64F)
#     sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
#     sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)

#     cv2.imshow('Original',frame)
#     cv2.imshow('Mask',mask)
#     cv2.imshow('laplacian',laplacian)
#     cv2.imshow('sobelx',sobelx)
#     cv2.imshow('sobely',sobely)

#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break


def findSignificantContour(edgeImg):
    contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)

        # From among them, find the contours with large surface area.
        contoursWithArea = []
        for tupl in level1Meta:
            contourIndex = tupl[0]
            contour = contours[contourIndex]
            area = cv2.contourArea(contour)
            contoursWithArea.append([contour, area, contourIndex])

    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour


while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)
    edges = cv2.Canny(frame, 100, 200)
    cv2.imshow('Edges', edges)

    contour = findSignificantContour(edges)
    # Draw the contour on the original image
    contourImg = np.copy(frame)
    cv2.drawContours(contourImg, [contour], 0,
                     (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    cv2.imshow('contourImg', contourImg)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
