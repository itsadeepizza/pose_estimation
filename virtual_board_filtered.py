from gesture_recognition.models.paolo.paolo_model import SVCModel

model = SVCModel()
model.load('gesture_recognition/models/paolo/svc.pkl')

import cv2
import concurrent.futures
import logging
import queue
import threading
import time
import numpy as np
import mediapipe as mp
import collections
from config import selected_config as conf
conf.set_derivate_parameters()


class KalmanFiltering():
    def __init__(self, r, q_p, q_v, p_p, p_v, delta_t):
        self.r = r
        self.q_p = q_p
        self.q_v = q_v
        self.p_p = p_p
        self.p_v = p_v


        self.R = np.diag([self.r, self.r])
        self.Q = np.diag([self.q_p, self.q_p, self.q_v, self.q_v])
        self.x_n = np.array([[0], [0], [0], [0]])

        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.P_n = np.diag([self.p_p, self.p_p, self.p_v, self.p_v])
        self.t_n = time.time()


    def update(self, z_n):
        delta_t = time.time() - self.t_n
        self.t_n = time.time()
        self.A = np.array([[1, 0, delta_t, 0],
                           [0, 1, 0, delta_t],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.x_p = self.A @ self.x_n
        self.P_p = self.A @ self.P_n @ self.A.transpose() + self.Q * delta_t
        self.K_n = self.P_p @ self.H.transpose() @ np.linalg.inv((self.H @ self.P_p @ self.H.transpose() + self.R))
        self.x_n = self.x_p + self.K_n @ (z_n - self.H @ self.x_p)
        self.P_n = self.P_p - self.K_n @ self.H @ self.P_p

    def last_position(self):
        return self.H @ self.x_n

    def infer_position(self):
        """Infer the position given the current time"""
        delta_t = time.time() - self.t_n
        last_position = self.last_position()
        last_speed = self.x_n[2:4]
        return last_position + last_speed * delta_t




class HMMFiltering():
    def __init__(self, trans_mat, alfa_0, freqs, model):
        """
        :param trans_mat: Transition matrix of the HMM
        :param alfa_0: Initial values for alpha_t
        :param freqs: Frequencies of the hidden states in model train dataset
        """
        self.trans_mat = trans_mat
        self.alfa_0 = alfa_0
        self.freqs = freqs
        self.alfa_t = self.alfa_0

    def update_alfa(self, unfiltered_probs):
        """Calculate alpha_t+1 given the observation y_t"""
        first_coeff = np.array(list(unfiltered_probs.values()) )/ self.freqs
        second_coeff = 0
        for i in range(len(self.alfa_t)):
            second_coeff += self.trans_mat[i] * self.alfa_t[i]
        self.alfa_t = first_coeff * second_coeff
        # Normalize alfa_t to avoid numerical errors
        self.alfa_t = self.alfa_t / sum(self.alfa_t)


    def predict_proba(self, unfiltered_probs):
        """Calculate the hidden state probability given the observation y_t"""
        classes = unfiltered_probs.keys()
        self.update_alfa(unfiltered_probs)
        return {gesture: prob for gesture, prob in zip(classes, self.alfa_t/sum(self.alfa_t))}



class MediaPipeProcessor():

    def __init__(self):

        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands          = mp.solutions.hands
        self.hands             = self.mp_hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)

        self.frame = None
        self.overlayer = None
        self.rigth_pointer_cords = collections.deque([None]*conf.LEN_QUEUE_MEDIAPIPE, maxlen=conf.LEN_QUEUE_MEDIAPIPE)
        self.rigth_gesture = collections.deque([None]*conf.LEN_QUEUE_MEDIAPIPE, maxlen=conf.LEN_QUEUE_MEDIAPIPE)
        self.left_pointer_cords = collections.deque([None]*conf.LEN_QUEUE_MEDIAPIPE, maxlen=conf.LEN_QUEUE_MEDIAPIPE)
        self.left_gesture = collections.deque([None]*conf.LEN_QUEUE_MEDIAPIPE, maxlen=conf.LEN_QUEUE_MEDIAPIPE)
        self.results = None

        # Graphic settings
        self.pencil_thickness = 15
        self.eraser_thickness = 200
        self.pencil_color = (1, 1, 1) # RGB, must be different from (0,0,0)
        self.max_distance = 1000 # Max admissible distance in pixel between two consecutive points

        self.pencil_color_BGR = (self.pencil_color[2], self.pencil_color[1], self.pencil_color[0])

        # Init the Kalman filtering
        self.kalman_filtering = {
            handeness: KalmanFiltering(r=2, q_p=10, q_v=200, p_p=10, p_v=10, delta_t=1 / 30) for handeness in ["Left", "Right"]
            }

        # Init the HMMmodel
        prob_trans = 0.01
        n = 5
        trans_mat = np.ones((n, n)) * prob_trans
        # Set the diagonal of trans_mat to 0.99
        trans_mat.flat[::n+1] = 1 - prob_trans * (n-1)


        self.hmm = {
            handeness: HMMFiltering(
                    trans_mat=trans_mat,
                    alfa_0=np.array([0] * (n - 1) + [1]),
                    freqs=np.array([0.2] * n),
                    model=model
                    )
            for handeness in ["Left", "Right"]
            }

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
        # Changes the color space in order to use Mediapipe
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.frame)
        # Changes the color space back to BGR for rendering with OpenCV
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        self.frame.flags.writeable = True

        # Update pointer cords and hand gestures
        self.update_pointers_cords()
        self.update_gestures()

        # Draw hands in AR
        self.draw_hands_landmarks()
        # Plot Board
        self.update_overlayer()
        # overlap overlayer to image
        self.frame = np.where(self.overlayer != 0, self.overlayer, self.frame)
        # Flip the image horizontally for a selfie-view display.
        self.frame = cv2.flip(self.frame, 1)

        return self.frame

    def get_pointer_cords(self, handeness='Right'):
        if self.results.multi_hand_landmarks:
            for hand_landmarks, multi_handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                # Check if it the handeness matches ( I need to reverse the condition otherwise it does not work ??)
                if multi_handedness.classification[0].label != handeness:
                    pointer_cords = np.array((int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.overlayer.shape[1]), int(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.overlayer.shape[0])))
                    self.kalman_filtering[handeness].update(pointer_cords.reshape(2, 1))

        filtered_position = self.kalman_filtering[handeness].infer_position().reshape(2).astype(int)
        return filtered_position

    def get_gesture(self, handeness='Right'):
        """Returns the gesture of the hand"""
        hand_detected = False
        if self.results.multi_hand_landmarks:
            for hand_landmarks, multi_handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                # Check if it the handeness matches ( I need to reverse the condition otherwise it does not work ??)
                if multi_handedness.classification[0].label != handeness:
                    # probs = {"open_hand": 0, "closed_hand": 0, "one": 0, "spiderman": 1}
                    unfiltered_probs = model.predict_proba(hand_landmarks, multi_handedness.classification[0].label)
                    unfiltered_probs["no_hand"] = 0
                    hand_detected = True
        if not hand_detected:
            unfiltered_probs = {"hand_closed": 0.05, "hand_opened": 0.05, "one": 0.05, "spiderman": 0.05, "no_hand": 0.8}
        filtered_probs = self.hmm[handeness].predict_proba(unfiltered_probs)
        # Return gesture with max probability
        detected_gesture = max(filtered_probs, key=filtered_probs.get)

        return detected_gesture

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
        if self.rigth_gesture[-1] == 'one' and self.rigth_gesture[-2] == 'one':
            # Only if the distance is not too long
            # TODO: if gesture != None then coordinates should be different from None too, maybe implement this in more robust way
            if np.linalg.norm(self.rigth_pointer_cords[-1] - self.rigth_pointer_cords[-2]) < 50:
                cv2.line(self.overlayer, self.rigth_pointer_cords[-2], self.rigth_pointer_cords[-1], self.pencil_color_BGR, self.pencil_thickness)

        # Left open hand to erase
        if self.left_gesture[-1] == 'hand_opened' and self.left_gesture[-2] == 'hand_opened':
            # Only if the distance is not too long
            # TODO: if gesture != None then coordinates should be different from None too, maybe implement this in more robust way
            if np.linalg.norm(self.left_pointer_cords[-1] - self.left_pointer_cords[-2]) < self.max_distance:
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
        tip_fingers_var = np.var(fingers_y) + np.var(fingers_x) + np.var(fingers_z)
        # Calculate std of n of the knuckles
        knuckles_x = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x])
        knuckles_y = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y])
        knuckles_z = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z])
        knuckles_var = np.var(knuckles_y) + np.var(knuckles_x) + np.var(knuckles_z)
        score_1 = np.sqrt(tip_fingers_var / knuckles_var)

        is_open = score_1 > 1
        return is_open


# bufferless VideoCapture
class VideoCapture:

  def __init__(self):
    self.cap = cv2.VideoCapture(conf.VIDEO_SOURCE)
    # Set opencv window size
    self.window_name = conf.WINDOW_NAME
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    if conf.RESOLUTION is not None:
        cv2.resizeWindow(self.window_name, conf.RESOLUTION[0], conf.RESOLUTION[1])
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
# source = 'http://192.168.1.54:8080/video'

conf.VIDEO_SOURCE = source
conf.RESOLUTION = None #(640, 480)


cap = VideoCapture()
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
    cv2.imshow(cap.window_name, frame)
    if chr(cv2.waitKey(1)&255) == 'q':
        break