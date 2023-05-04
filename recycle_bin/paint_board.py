import cv2
import mediapipe as mp
import random

import numpy
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


board = None
is_right = False
score_1= 0
idx = 0
# For webcam input:
cap = cv2.VideoCapture(0)

list_strokes = [[]]
is_open = True
with mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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

        if board is None:
            # convert image to numpy array
            image = np.array(image)
            # Create a transparent image for opencv
            board = np.zeros(image.shape, np.uint8)



        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if results.multi_hand_landmarks:
            for hand_landmarks, multi_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Check if it is a right hand
                is_right = multi_handedness.classification[0].label != 'Right'
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

                # Right hand to draw
                if is_right:
                    # Calculate std of tip of the fingers
                    fingers_x = numpy.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x])
                    fingers_y = numpy.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
                    fingers_z = numpy.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z])
                    tip_fingers_std = np.std(fingers_y) + np.std(fingers_x) + np.std(fingers_z)
                    # Calculate std of n of the knuckles
                    knuckles_x = numpy.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x])
                    knuckles_y = numpy.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y])
                    knuckles_z = numpy.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z])
                    knuckles_std = np.std(knuckles_y) + np.std(knuckles_x) + np.std(knuckles_z)
                    score_1 = tip_fingers_std / knuckles_std
                    pointer_cords = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]))
                    thickness = 3
                    # Add pixel to board
                    # board[pointer_cords[1] - thickness:pointer_cords[1]+thickness, pointer_cords[0]-thickness:pointer_cords[0]+thickness] = (255, 255, 255)


                    is_open = score_1 > 0.7
                    current_stroke = list_strokes[-1]
                    if not is_open:
                        current_stroke.append(pointer_cords)
                    elif len(current_stroke) > 0:
                        list_strokes.append([])
                # Left hand to erase
                else:
                    eraser_cords = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]))
                    eraser_thickess = 10
                    # Broke all lines that are close to the eraser
                    new_list_strokes = []
                    for stroke in list_strokes:
                        new_stroke = []
                        for i in range(len(stroke) - 1):
                            new_stroke.append(stroke[i])
                            line = np.array(stroke[i:i + 2])
                            # CAlculate distance from line to point
                            dist = min(np.linalg.norm(eraser_cords - line[0]), np.linalg.norm(eraser_cords - line[1]))
                            if dist < eraser_thickess:
                                # broke line
                                new_list_strokes.append(new_stroke)
                                new_stroke = []
                            new_stroke.append(stroke[i + 1])
                        new_list_strokes.append(new_stroke)
                    list_strokes = new_list_strokes
        for stroke in list_strokes:
            if len(stroke) > 0:
                cv2.polylines(image, [np.array(stroke)], False, (255, 255, 255), 3)
        # overlap board to image
        # image = cv2.addWeighted(image, 0.5, board, 0.5, 0)
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)

        image = cv2.putText(image, f"{score_1:.2f}  {str(is_open)} {str(is_right)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
