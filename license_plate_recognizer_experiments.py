from absl import app
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from pytesseract import Output
import imutils

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def segment_1(img):
    h, w, c = img.shape
    boxes = pytesseract.image_to_boxes(img)
    print('segmentation1 result boxes: ' + str(len(boxes)))
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0,255,255), 2)

    cv2.imshow('img', img)
    cv2.waitKey()

def segment_2(img):
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['text'])
    print('segmentation2 result boxes: ' + str(n_boxes))
    for i in range(n_boxes):
        if int(float(d['conf'][i])) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)

def recognize_license_plate():

    img_plate = cv2.imread('./data/license-plates/lp.jpg')
    img_plate = cv2.imread('./data/license-plates/1clean.png')
    plt.imshow(cv2.cvtColor(img_plate, cv2.COLOR_BGR2RGB))
    plt.show()

    gray = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    threshold = 50
    thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]

    plt.imshow(thresh, cmap='gray')
    plt.title('Thresholded');
    plt.show()

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cropped_image = []

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            if w < 50: continue  # exclude small contours

            imageCopy = image.copy()
            cv2.drawContours(imageCopy, [c], -1, (0, 255, 0), 2)

            plt.imshow(cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB))
            plt.title('Contour');
            plt.show()

            cropped_image = image[y:y + h, x:x + w]
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.title('Cropped');
            plt.show()

            ar = w / float(h)
            if ar > 1.3: break  # for square license plates

    #img_plate = opening(img_plate)
    # cv2.imshow('img', img_plate)
    # cv2.waitKey(0)
    #cropped_image = img_plate
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        imageCopy = cropped_image.copy()

        moments = cv2.moments(c)
        huMoments = cv2.HuMoments(moments)
        h0, h1, h2 = huMoments[:3, 0]

        area = int(cv2.contourArea(c))

        if area < 100: continue

        print(f'{h0},{h1},{h2}')

        cv2.drawContours(imageCopy, [c], -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB))
        plt.title('Contour');
        plt.show()







    # result = pytesseract.image_to_string(img_plate)
    # mytext = re.sub(r"[^\w]", '', result)
    # print('risultato targa:' + mytext)



def main(_argv):
    recognize_license_plate()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass