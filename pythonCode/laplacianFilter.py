import cv2
import sys
import numpy as np

#read image
image = cv2.imread("../assets/crop1.png", cv2.IMREAD_GRAYSCALE)

#check if image exists
if image is None:
    print("can not find image")
    sys.exit()

#apply laplacian
laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3, scale=1, delta=0)

#create lop kernel
logKernel = np.array(( [0.4038, 0.8021, 0.4038], [0.8021, -4.8233, 0.8021], [0.4038, 0.8021, 0.4038]), dtype="float")

#filter image using log kernel
logimg = cv2.filter2D(image, cv2.CV_32F, logKernel)

#normalize images
cv2.normalize(laplacian, laplacian, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.normalize(logimg, logimg, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#create windows to display images
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.namedWindow("laplacian", cv2.WINDOW_NORMAL)
cv2.namedWindow("log", cv2.WINDOW_NORMAL)

#display images
cv2.imshow("image", image)
cv2.imshow("laplacian", laplacian)
cv2.imshow("log", logimg)

#press esc to exit the program
cv2.waitKey(0)

#close all the opened windows
cv2.destroyAllWindows()