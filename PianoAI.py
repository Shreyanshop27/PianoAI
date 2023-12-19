import cv2
import pytesseract
import mediapipe as mp
from pydub import AudioSegment
import pygame

# Initialize Pygame mixer
pygame.mixer.init()



# Set tesseract path to the location where it's installed
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path

# Define config parameters for tesseract
config = ('-l eng --oem 1 --psm 3')

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks

# Load the audio file
do = pygame.mixer.Sound(r'Notes\do.mp3')
re = pygame.mixer.Sound(r'Notes\re.mp3')
mi = pygame.mixer.Sound(r'Notes\mi.mp3')
fa = pygame.mixer.Sound(r'Notes\fa.mp3')
sol = pygame.mixer.Sound(r'Notes\sol.mp3')
la = pygame.mixer.Sound(r'Notes\la.mp3')

# Start capturing video 
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image from OpenCV BGR format to MediaPipe RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform hand landmark detection
    results = hands.process(frame_rgb)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Perform text detection
    d = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT, config=config)

    # Draw bounding boxes around text and check for overlap with fingertips
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        print(d['text'][i])
        if d['text'][i] in 'do':
            # Increase the size of the bounding box by an offset
            offset = 20  # Change this value as needed
            (x, y, w, h) = (d['left'][i] - offset, d['top'][i] - offset, d['width'][i] + 2*offset, d['height'][i] + 2*offset)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if any fingertip overlaps with this bounding box
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the coordinates of the index fingertip
                    fingertip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    fingertip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                    # Check if the fingertip overlaps with the bounding box
                    if x <= fingertip_x <= x + w and y <= fingertip_y <= y + h:
                        # If the text is 'C4', play the sound
                        do.play()
        if d['text'][i] in 're':
            # Increase the size of the bounding box by an offset
            offset = 20  # Change this value as needed
            (x, y, w, h) = (d['left'][i] - offset, d['top'][i] - offset, d['width'][i] + 2*offset, d['height'][i] + 2*offset)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if any fingertip overlaps with this bounding box
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the coordinates of the index fingertip
                    fingertip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    fingertip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                    # Check if the fingertip overlaps with the bounding box
                    if x <= fingertip_x <= x + w and y <= fingertip_y <= y + h:
                        # If the text is 'C4', play the sound
                        re.play()
        if d['text'][i] in 'mi':
            # Increase the size of the bounding box by an offset
            offset = 20  # Change this value as needed
            (x, y, w, h) = (d['left'][i] - offset, d['top'][i] - offset, d['width'][i] + 2*offset, d['height'][i] + 2*offset)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if any fingertip overlaps with this bounding box
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the coordinates of the index fingertip
                    fingertip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    fingertip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                    # Check if the fingertip overlaps with the bounding box
                    if x <= fingertip_x <= x + w and y <= fingertip_y <= y + h:
                        # If the text is 'C4', play the sound
                        mi.play()
        if d['text'][i] in 'fa':
            # Increase the size of the bounding box by an offset
            offset = 20  # Change this value as needed
            (x, y, w, h) = (d['left'][i] - offset, d['top'][i] - offset, d['width'][i] + 2*offset, d['height'][i] + 2*offset)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if any fingertip overlaps with this bounding box
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the coordinates of the index fingertip
                    fingertip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    fingertip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                    # Check if the fingertip overlaps with the bounding box
                    if x <= fingertip_x <= x + w and y <= fingertip_y <= y + h:
                        # If the text is 'C4', play the sound
                        fa.play()
        if d['text'][i] in 'sol':
            # Increase the size of the bounding box by an offset
            offset = 20  # Change this value as needed
            (x, y, w, h) = (d['left'][i] - offset, d['top'][i] - offset, d['width'][i] + 2*offset, d['height'][i] + 2*offset)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if any fingertip overlaps with this bounding box
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the coordinates of the index fingertip
                    fingertip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    fingertip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                    # Check if the fingertip overlaps with the bounding box
                    if x <= fingertip_x <= x + w and y <= fingertip_y <= y + h:
                        # If the text is 'C4', play the sound
                        sol.play()
        if d['text'][i] in 'la':
            # Increase the size of the bounding box by an offset
            offset = 20  # Change this value as needed
            (x, y, w, h) = (d['left'][i] - offset, d['top'][i] - offset, d['width'][i] + 2*offset, d['height'][i] + 2*offset)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if any fingertip overlaps with this bounding box
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the coordinates of the index fingertip
                    fingertip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    fingertip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                    # Check if the fingertip overlaps with the bounding box
                    if x <= fingertip_x <= x + w and y <= fingertip_y <= y + h:
                        # If the text is 'C4', play the sound
                        la.play()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
