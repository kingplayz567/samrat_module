import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread("Screenshot 2022-09-09 180214.png")
cap = cv.VideoCapture('video.mp4')
img2 = cv.imread("Animal_database_flowchart_home.jpg")
def read_video():
    while True:
        isTrue, frame = cap.read()
        cv.imshow("Video", frame)

        if cv.waitKey(20) & 0xff == ord('d'):
            break
    cap.release()
    cv.destroyAllWindows()

def translation():
    height, width = img.shape[:2]
    quarter_height, quarter_width = height / 4, width / 4
    T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
    img_translation = cv.warpAffine(img, T, (width, height))
    cv.imshow("Originalimage", img)
    cv.imshow('Translation', img_translation)
    cv.waitKey()
    cv.destroyAllWindows()

def contours():
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edged = cv.Canny(gray, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.imshow('Canny Edges After Contouring', edged)
    cv.waitKey(0)
    print("Number of Contours found = " + str(len(contours)))
    cv.drawContours(img, contours, -1, (0, 255, 0), 1)
    cv.imshow('Contours', img)
    cv.waitKey()
    cv.destroyAllWindows()

def rotation():
    img = cv.imread('Screenshot 2022-09-09 180214.png', 0)
    window_name = 'Image'
    image = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    cv.imshow(window_name, image)
    cv.waitKey(0)

def scaling():
    img = cv.imread('Screenshot 2022-09-09 180214.png', 0)
    rows, cols = img.shape
    img_shrinked = cv.resize(img, (250, 200),
                             interpolation=cv.INTER_AREA)
    cv.imshow('img', img_shrinked)
    cv.waitKey(0)
    img_enlarged = cv.resize(img_shrinked, None,
                             fx=1.5, fy=1.5,
                             interpolation=cv.INTER_CUBIC)
    cv.imshow('img', img_enlarged)
    cv.waitKey(0)
    cv.destroyAllWindows()
def reflection():
    img = cv.imread('Screenshot 2022-09-09 180214.png', 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
    reflected_img = cv.warpPerspective(img, M, (int(cols), int(rows)))
    cv.imshow('img', reflected_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def bitwise_and():
    img1 = cv.imread('Screenshot 2022-09-09 180214.png')
    img2 = cv.imread('reflection_out.jpg')
    dest_and = cv.bitwise_and(img2, img1, mask=None)
    cv.imshow('Bitwise And', dest_and)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def bitwise_or():
    img1 = cv.imread('Screenshot 2022-09-09 180214.png')
    img2 = cv.imread('reflection_out.jpg')
    dest_or = cv.bitwise_or(img2, img1, mask=None)
    cv.imshow('Bitwise OR', dest_or)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
def bitwise_xor():
    img1 = cv.imread('Screenshot 2022-09-09 180214.png')
    img2 = cv.imread('reflection_out.jpg')
    dest_xor = cv.bitwise_xor(img1, img2, mask=None)
    cv.imshow('Bitwise XOR', dest_xor)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()
def bitwise_not():
    img1 = cv.imread('Screenshot 2022-09-09 180214.png')
    img2 = cv.imread('reflection_out.jpg')
    dest_not1 = cv.bitwise_not(img1, mask=None)
    dest_not2 = cv.bitwise_not(img2, mask=None)
    cv.imshow('Bitwise NOT on image 1', dest_not1)
    cv.imshow('Bitwise NOT on image 2', dest_not2)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def crop():
    crop = img[50:180, 100:300]
    cv.imshow('original', img)
    cv.imshow('cropped', crop)
    cv.waitKey(0)
    cv.destroyAllWindows()

def shearing_x():
    rows, cols = img.shape
    M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    sheared_img = cv.warpPerspective(img, M, (int(cols * 1.5), int(rows * 1.5)))
    cv.imshow('img', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def shearing_y():
    rows, cols = img.shape
    M = np.float32([[1, 0, 0], [0.5, 1, 0], [0, 0, 1]])
    sheared_img = cv.warpPerspective(img, M, (int(cols * 1.5), int(rows * 1.5)))
    cv.imshow('sheared_y-axis_out.jpg', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def functions():
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)

    canny = cv.Canny(img, 525, 525)
    cv.imshow("Canny", canny)

    dilated = cv.dilate(canny, (3, 3), iterations=1)
    cv.imshow("dilated", dilated)

    eroded = cv.erode(dilated, (3, 3), iterations=1)
    cv.imshow('Erode', eroded)
def masking():
    blank = np.zeros(img.shape[:2], dtype='uint8')
    cv.imshow('Blank Image', blank)
    circle = cv.circle(blank,
                       (img.shape[1] // 2, img.shape[0] // 2), 200, 255, -1)
    cv.imshow('Mask', circle)
    masked = cv.bitwise_and(img, img, mask=circle)
    cv.imshow('Masked Image', masked)
    cv.waitKey(0)

def alpha_blending():
    img1 = cv.imread("Screenshot 2022-09-09 180214.png")
    img2 = cv.imread("Animal_database_flowchart_home.jpg")
    img2 = cv.resize(img2, img1.shape[1::-1])
    cv.imshow("img 1", img1)
    cv.waitKey(0)
    cv.imshow("img 2", img2)
    cv.waitKey(0)
    choice = 1
    while(choice):
        alpha = float(input("Enter alpha value"))
    dst = cv.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    cv.imwrite('alpha_mask_.png', dst)
    img3 = cv.imread('alpha_mask_.png')
    cv.imshow("alpha blending 1", img3)
    cv.waitKey(0)
    choice = int(input("Enter 1 to continue and 0 to exit"))

def histogram():
    histr = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(histr)
    plt.show()

def kernel_blur():
    kernel2 = np.ones((5, 5), np.float32) / 25
    img = cv.filter2D(src = img2, ddepth=-1, kernel=kernel2)
    cv.imshow('Original', img2)
    cv.imshow('Kernel Blur', img)
    cv.waitKey()
    cv.destroyAllWindows()

def avg_blur():
    averageBlur = cv.blur(img, (5, 5))
    cv.imshow('Original', img)
    cv.imshow('Average blur', averageBlur)
    cv.waitKey()
    cv.destroyAllWindows()

def gaussian_blur():
    gaussian = cv.GaussianBlur(img, (3, 3), 0)
    cv.imshow('Original', img)
    cv.imshow('Gaussian blur', gaussian)
    cv.waitKey()
    cv.destroyAllWindows()

def median_blur():
    median = cv.medianBlur(img, 9)
    cv.imshow('Original', img)
    cv.imshow('Median blur', median)
    cv.waitKey()
    cv.destroyAllWindows()

def bilateral_blur():
    bilateral = cv.bilateralFilter(img, 9, 75, 75)
    cv.imshow('Original', img)
    cv.imshow('Bilateral blur', bilateral)
    cv.waitKey()
    cv.destroyAllWindows()