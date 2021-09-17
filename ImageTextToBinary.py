import cv2
import pytesseract
import numpy as np


# Initialize Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# add image and convert
img = cv2.imread("Resources/res1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create separate(second) translated text image
img2 = np.zeros((512, 512, 3), np.uint8)

# add third image and convert
img3 = img.copy()



def toBinary(a):
    l, m = [], []
    for i in a:
        l.append(ord(i))
    for i in l:
        m.append(int(bin(i)[2:]))
    return m


# Run translation
words = pytesseract.image_to_string(img)
result = (toBinary(words))
listTo = ' '.join([str(elem) for elem in result])

print('Image Source Text:', words)

# Detecting Words
hImg, wImg, _ = img.shape
boxes = pytesseract.image_to_data(img)

# Loop to create bounding box over each word in source text
for x, b in enumerate(boxes.splitlines()):

    if x != 0:
        b = b.split()
        if len(b) == 12:
            # Dimensions for each word
            x, y, w, h, = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            # Create bounding box around each word
            cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 1)
            # Add translated text to image.
            # cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, .25, (50, 50, 255), 1)

print('Converted Text: ', listTo)

# Create copy of original image with source text above each word
for x, c in enumerate(boxes.splitlines()):
    if x != 0:
        c = c.split()
        if len(c) == 12:
            x, y, w, h = int(c[6]), int(c[7]), int(c[8]), int(c[9])
            cv2.rectangle(img3, (x, y), (w + x, h + y), (0, 0, 255), 1)
            # Add translated text to image.
            cv2.putText(img3, c[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

# Add translated text to second image.
cv2.putText(img2, listTo, (20, 170), cv2.FONT_HERSHEY_COMPLEX, .4, (0, 255, 0), 2)

# Add blur to original image
imgBlur = cv2.GaussianBlur(img, (7, 7), 0)  # must be odd numbers for ksize

imgResize = cv2.resize(imgBlur, (1000, 500))
imgResize2 = cv2.resize(img2, (1000, 500))
imgResize3 = cv2.resize(img3, (1000, 500))

# Apply Blending Function
alpha = 0.5
beta = (1.0 - alpha)
translatedBlend = cv2.addWeighted(imgResize, alpha, imgResize2, beta, 0.0)


cv2.imshow('Translated', translatedBlend)
cv2.imshow('Original', imgResize3)
cv2.waitKey(0)
cv2.destroyAllWindows()
