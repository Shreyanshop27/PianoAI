import cv2
import pytesseract

# Set tesseract path to the location where it's installed
# pytesseract.pytesseract.tesseract_cmd = r''  # Update this path

# Define config parameters for tesseract
config = ('-l eng --oem 1 --psm 3')

# Start capturing video 
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image from OpenCV BGR format to Tesseract RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform text detection
    d = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT, config=config)

    # Draw bounding boxes around text
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert the image from Tesseract RGB format to OpenCV BGR format
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
