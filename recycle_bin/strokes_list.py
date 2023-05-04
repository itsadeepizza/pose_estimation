import cv2
import mediapipe as mp
import random
import time
import threading
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


overlayer = None
is_right = False
score_1= 0
idx = 0

# Use a phone ip camera on local address as input
# cap = cv2.VideoCapture('http://192.168.1.54:8080/video')


# For webcam input:
cap = cv2.VideoCapture(0)
last_pointer_cords = None


# Set opencv window size
cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Hands', 1728, 922)

# Create two threads: one for capturing image and one for image processing
# t1 = threading.Thread(target=capture_image)


def is_hand_open(hand_landmarks):
    """Check if the hand is open or closed"""
    # Calculate std of tip of the fingers
    fingers_x = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x])
    fingers_y = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
    fingers_z = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z])
    tip_fingers_std = np.std(fingers_y) + np.std(fingers_x) + np.std(fingers_z)
    # Calculate std of n of the knuckles
    knuckles_x = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x])
    knuckles_y = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y])
    knuckles_z = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z])
    knuckles_std = np.std(knuckles_y) + np.std(knuckles_x) + np.std(knuckles_z)
    score_1 = tip_fingers_std / knuckles_std


    is_open = score_1 > 0.9
    return is_open


def update_overlayer(overlayer, results, last_pointer_cords):
    """Update overlayer with the hand strokes"""
    pointer_cords = last_pointer_cords
    if results.multi_hand_landmarks:
        for hand_landmarks, multi_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Check if it is a right hand
            is_right = multi_handedness.classification[0].label != 'Right'

            # Right hand to draw
            if is_right:
                pointer_cords = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * overlayer.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * overlayer.shape[0])))
                thickness = 3
                is_open = is_hand_open(hand_landmarks)
                # Add stroke to board
                if not is_open:
                    if last_pointer_cords is not None and np.linalg.norm(pointer_cords - last_pointer_cords) < 50:
                        cv2.line(overlayer, last_pointer_cords, pointer_cords, (255, 255, 255), thickness)
                else:
                    pointer_cords = None


            # Left hand to erase
            else:
                eraser_cords = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * overlayer.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * overlayer.shape[0]))
                eraser_thickess = 30
                # Erase all points near eraser_cords
                cv2.circle(overlayer, eraser_cords, eraser_thickess, (0, 0, 0), -1)

            # Add the hand landmarks to the image
            # mp_drawing.draw_landmarks(overlayer, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
    return overlayer, pointer_cords


pointer_cords = None
is_open = True
with mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as hands:
    while cap.isOpened():
        idx = (idx + 1) % 1
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue


        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if overlayer is None:
            # convert image to np array
            image = np.array(image)
            # Create a transparent image for opencv
            overlayer = np.zeros(image.shape, np.uint8)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Update overlayer
        overlayer, pointer_cords = update_overlayer(overlayer, results, pointer_cords)

        # overlap overlayer to image
        image = cv2.addWeighted(image, 0.5, overlayer, 0.5, 0)
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)

        image = cv2.putText(image, f"{score_1:.2f}  {str(is_open)} {str(is_right)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
