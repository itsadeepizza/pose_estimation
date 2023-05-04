from models.paolo.paolo_model import SVCModel

model = SVCModel()
model.load('models/paolo/svc.pkl')

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

# Set size of text and bars on the overlay
ui_size = 40 #px

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
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False)

        self.frame = None
        self.overlayer = None
        self.rigth_pointer_cords = collections.deque([None]*conf.LEN_QUEUE_MEDIAPIPE, maxlen=conf.LEN_QUEUE_MEDIAPIPE)
        self.rigth_gesture = collections.deque([None]*conf.LEN_QUEUE_MEDIAPIPE, maxlen=conf.LEN_QUEUE_MEDIAPIPE)
        self.left_pointer_cords = collections.deque([None]*conf.LEN_QUEUE_MEDIAPIPE, maxlen=conf.LEN_QUEUE_MEDIAPIPE)
        self.left_gesture = collections.deque([None]*conf.LEN_QUEUE_MEDIAPIPE, maxlen=conf.LEN_QUEUE_MEDIAPIPE)
        self.results = None

        # Graphic settings
        self.pencil_thickness = 3
        self.eraser_thickness = 50

        # Init the HMMmodel
        prob_trans = 0.01
        n = 5
        trans_mat = np.ones((n, n)) * prob_trans
        # Set the diagonal of trans_mat to 0.99
        trans_mat.flat[::n+1] = 1 - prob_trans * (n-1)

        self.hmm = HMMFiltering(
            trans_mat=trans_mat,
            alfa_0=np.array([0] * (n - 1) + [1]),
            freqs=np.array([0.2] * n),
            model=model
            )

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

        # Flip the image horizontally for a selfie-view display.
        self.frame = cv2.flip(self.frame, 1)
        # Plot gesture recognition
        self.update_overlayer()


        # Flip the image horizontally for a selfie-view display.

        # overlap overlayer to image
        self.frame = np.where(self.overlayer != 0, self.overlayer, self.frame)
        # self.frame = cv2.addWeighted(self.frame, 0.5, self.overlayer, 0.5, 0)
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
        filtered_probs = self.hmm.predict_proba(unfiltered_probs)
        return filtered_probs, unfiltered_probs

    def update_pointers_cords(self):
        self.left_pointer_cords.append(self.get_pointer_cords('Left'))
        self.rigth_pointer_cords.append(self.get_pointer_cords('Right'))

    def update_gestures(self):
        # self.left_gesture.append(self.get_gesture('Left'))
        self.rigth_gesture.append(self.get_gesture('Right'))

    def draw_hands_landmarks(self):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def update_overlayer(self):
        """Update overlayer with the hand strokes"""
        self.overlayer = np.zeros(self.overlayer.shape, np.uint8)
        # Create an empty overlayer
        unfiltered_layer = np.zeros(self.overlayer.shape, np.uint8)

        filtered_probs = self.rigth_gesture[-1][0]
        unfiltered_probs = self.rigth_gesture[-1][1]
        print(filtered_probs)
        # Show gesture probabilities as gauge bars
        if filtered_probs is not None:
            for i, (gesture, prob) in enumerate(filtered_probs.items()):
                cv2.rectangle(self.overlayer, (0, i*ui_size), (int(prob*ui_size*5), (i+1)*ui_size), (255, 0, 0), -1)

            for i, (gesture, prob) in enumerate(unfiltered_probs.items()):
                cv2.rectangle(self.overlayer, (0, i*ui_size), (int(prob*ui_size*5), (i+1)*ui_size), (0, 255, 0), -1)
                cv2.putText(self.overlayer, gesture, (0, (i+1)*ui_size), cv2.FONT_HERSHEY_SIMPLEX, ui_size/40, (255, 255, 255), int(ui_size/20), cv2.LINE_AA)
        # Add a legend with green and blue squares at theright of the screen
        cv2.rectangle(self.overlayer, (ui_size * 6, 0), (ui_size * 7, ui_size), (0, 255, 0), -1)
        cv2.putText(self.overlayer, "Unfiltered", (int(ui_size * 7.5), ui_size), cv2.FONT_HERSHEY_SIMPLEX, ui_size/40, (255, 255, 255), int(ui_size/20), cv2.LINE_AA)
        cv2.rectangle(self.overlayer, (ui_size * 6, int(ui_size * 1.5)), (ui_size * 7, int(ui_size * 2.5)), (255, 0, 0), -1)
        cv2.putText(self.overlayer, "Filtered", (int(ui_size * 7.5), int(ui_size * 2.5)), cv2.FONT_HERSHEY_SIMPLEX, ui_size/40, (255, 255, 255), int(ui_size/20), cv2.LINE_AA)

        #Overlap unfiltered probs to overlayer
        # self.overlayer = cv2.addWeighted(self.overlayer, 0.5, unfiltered_layer, 0.5, 0.5)


class VideoCapture:
  def __init__(self):
    self.cap = cv2.VideoCapture(conf.VIDEO_SOURCE)
    # Set opencv window size
    self.window_name = conf.WINDOW_NAME
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
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
source = 'http://192.168.1.54:8080/video'

conf.VIDEO_SOURCE = source
conf.RESOLUTION = (640, 480)


cap       = VideoCapture()
processor = MediaPipeProcessor()
# Deque to store timestamps of last 10 frames in a circular array
frame_timestamps = collections.deque(maxlen=10)
while True:
    frame_timestamps.append(time.time())
    # calculate the fps
    fps = len(frame_timestamps) / (frame_timestamps[-1] - frame_timestamps[0] + 0.0001)
    frame = cap.read()
    frame = processor.process(frame)

    cv2.putText(frame, "FPS: {:.2f}".format(fps), (frame.shape[1]-ui_size*5, ui_size), cv2.FONT_HERSHEY_SIMPLEX, int(ui_size/40), (0, 255, 0), int(ui_size/20))
    cv2.imshow(cap.window_name, frame)
    if chr(cv2.waitKey(1)&255) == 'q':
        break