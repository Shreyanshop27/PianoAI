import cv2
import pytesseract
import numpy as np

# Set tesseract path to the location where it's installed
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path

# Define config parameters for tesseract
config = ('-l eng --oem 1 --psm 3')

# Start capturing video 
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Apply noise removal
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(thresh, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Perform text detection
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)

    # Draw bounding boxes around text
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
