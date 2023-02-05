import cv2
import concurrent.futures
import logging
import queue
import threading
import time
import numpy as np
import mediapipe as mp
import collections


class MediaPipeProcessor():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)

        self.frame = None
        self.overlayer = None
        self.rigth_pointer_cords = collections.deque([None]*10, maxlen=10)
        self.rigth_gesture = collections.deque([None]*10, maxlen=10)
        self.left_pointer_cords = collections.deque([None]*10, maxlen=10)
        self.left_gesture = collections.deque([None]*10, maxlen=10)
        self.results = None

        # Graphic settings
        self.pencil_thickness = 3
        self.eraser_thickness = 50

    def process(self, frame):
        self.frame = frame

        # Init overlayer if not initialized
        if self.overlayer is None:
            # convert image to np array
            image = np.array(frame)
            # Create a transparent image for opencv
            self.overlayer = np.zeros(image.shape, np.uint8)

        # To improve performance, optionally mark the image as not writeable to pass by reference
        self.frame.flags.writeable = False
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.frame)
        self.frame.flags.writeable = True

        # Update pointer cords and hand gestures
        self.update_pointers_cords()
        self.update_gestures()

        # Update overlayer draws
        self.update_overlayer()

        # Draw hands in AR
        self.draw_hands_landmarks()

        # overlap overlayer to image
        self.frame = cv2.addWeighted(self.frame, 0.5, self.overlayer, 0.5, 0)
        # Flip the image horizontally for a selfie-view display.
        self.frame = cv2.flip(self.frame, 1)

        return self.frame

    def get_pointer_cords(self, handeness='Right'):
        if self.results.multi_hand_landmarks:
            for hand_landmarks, multi_handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                # Check if it the handeness matches ( I need to reverse the condition otherwise it does not work ??)
                if multi_handedness.classification[0].label != handeness:
                    pointer_cords = np.array((int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.overlayer.shape[1]), int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.overlayer.shape[0])))
                    return pointer_cords
        return None

    def get_gesture(self, handeness='Right'):
        if self.results.multi_hand_landmarks:
            for hand_landmarks, multi_handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                # Check if it the handeness matches ( I need to reverse the condition otherwise it does not work ??)
                if multi_handedness.classification[0].label != handeness:
                    is_open = MediaPipeProcessor.is_hand_open(hand_landmarks)
                    if is_open:
                        return 'Open'
                    else:
                        return 'Closed'
        return None

    def update_pointers_cords(self):
        self.left_pointer_cords.append(self.get_pointer_cords('Left'))
        self.rigth_pointer_cords.append(self.get_pointer_cords('Right'))

    def update_gestures(self):
        self.left_gesture.append(self.get_gesture('Left'))
        self.rigth_gesture.append(self.get_gesture('Right'))

    def draw_hands_landmarks(self):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def update_overlayer(self):
        """Update overlayer with the hand strokes"""

        # Right closed hand to draw
        # Only if the hand was closed in two last frames
        if self.rigth_gesture[-1] == 'Closed' and self.rigth_gesture[-2] == 'Closed':
            # Only if the distance is not too long
            # TODO: if gesture != None then coordinates should be different from None too, maybe implement this in more robust way
            if np.linalg.norm(self.rigth_pointer_cords[-1] - self.rigth_pointer_cords[-2]) < 50:
                cv2.line(self.overlayer, self.rigth_pointer_cords[-2], self.rigth_pointer_cords[-1], (255, 255, 255), self.pencil_thickness)

        # Left open hand to erase
        if self.left_gesture[-1] == 'Open' and self.left_gesture[-2] == 'Open':
            # Only if the distance is not too long
            # TODO: if gesture != None then coordinates should be different from None too, maybe implement this in more robust way
            if np.linalg.norm(self.left_pointer_cords[-1] - self.left_pointer_cords[-2]) < 200:
                # Erase points using a line
                cv2.line(self.overlayer, self.left_pointer_cords[-2], self.left_pointer_cords[-1], (0, 0, 0), self.eraser_thickness, -1)


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

# Webcam laptop
source = 0
# Webcam phone ipcam
source = 'http://192.168.1.54:8080/video'


cap = VideoCapture(source)
processor = MediaPipeProcessor()
# Deque to store timestamps of last 10 frames in a circular array
frame_timestamps = collections.deque(maxlen=10)
while True:
    frame_timestamps.append(time.time())
    # calculate the fps
    fps = len(frame_timestamps) / (frame_timestamps[-1] - frame_timestamps[0] + 0.0001)
    frame = cap.read()
    frame = processor.process(frame)

    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("frame", frame)
    if chr(cv2.waitKey(1)&255) == 'q':
        break