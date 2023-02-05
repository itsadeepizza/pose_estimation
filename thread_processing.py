import cv2
import concurrent.futures
import logging
import queue
import threading
import time
import numpy as np
import mediapipe as mp


class MediaPipeProcessor():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)

        self.frame = None
        self.overlayer = None
        self.pointer_cords = None
        self.last_pointer_cords = None
        self.results = None

    def process(self, frame):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference
        self.frame = frame
        if self.overlayer is None:
            # convert image to np array
            image = np.array(frame)
            # Create a transparent image for opencv
            self.overlayer = np.zeros(image.shape, np.uint8)

        self.frame.flags.writeable = False
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.frame)
        self.frame.flags.writeable = True
        self.update_overlayer()

        # overlap overlayer to image
        self.frame = cv2.addWeighted(self.frame, 0.5, self.overlayer, 0.5, 0)
        # Flip the image horizontally for a selfie-view display.
        self.frame = cv2.flip(self.frame, 1)

        return self.frame

    def update_overlayer(self):
        """Update overlayer with the hand strokes"""
        if self.results.multi_hand_landmarks:
            for hand_landmarks, multi_handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                # Check if it is a right hand
                is_right = multi_handedness.classification[0].label != 'Right'

                # Right hand to draw
                if is_right:
                    pointer_cords = np.array((int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.overlayer.shape[1]), int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.overlayer.shape[0])))
                    thickness = 3
                    is_open = MediaPipeProcessor.is_hand_open(hand_landmarks)
                    # Add stroke to board
                    if not is_open:
                        if self.last_pointer_cords is not None and np.linalg.norm(pointer_cords - self.last_pointer_cords) < 50:
                            cv2.line(self.overlayer, self.last_pointer_cords, pointer_cords, (255, 255, 255), thickness)
                    else:
                        pointer_cords = None
                    self.last_pointer_cords = pointer_cords


                # Left hand to erase
                else:
                    eraser_cords = (int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.overlayer.shape[1]), int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.overlayer.shape[0]))
                    eraser_thickess = 30
                    # Erase all points near eraser_cords
                    cv2.circle(self.overlayer, eraser_cords, eraser_thickess, (0, 0, 0), -1)

                # Add the hand landmarks to the image
                self.mp_drawing.draw_landmarks(self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.mp_drawing_styles.get_default_hand_landmarks_style(), self.mp_drawing_styles.get_default_hand_connections_style())

    @staticmethod
    def is_hand_open(hand_landmarks):
        """Check if the hand is open or closed"""
        # Calculate std of tip of the fingers
        mp_hands = mp.solutions.hands
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


# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


cap = VideoCapture('http://192.168.1.54:8080/video')
processor = MediaPipeProcessor()
frame_timestamps = []
while True:
    frame_timestamps.append(time.time())
    # Remove the oldest frame timestamp if we're storing too many
    if len(frame_timestamps) > 10:
        frame_timestamps.pop(0)
    # calculate the fps
    fps = len(frame_timestamps) / (frame_timestamps[-1] - frame_timestamps[0] + 0.0001)
    frame = cap.read()
    frame = processor.process(frame)

    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    if chr(cv2.waitKey(1)&255) == 'q':
        break