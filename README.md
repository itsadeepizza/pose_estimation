# A virtual whiteboard with hand gestures recognition

This project is an implementation of a virtual whiteboard using hand gestures recognition. The project is based on the MediaPipe library and is entirely written in Python.
You write with the right hand, using the "one" gesture, and you can erase with the left hand, using an open hand.



## Structure

The project is structured as follows:

`gesture_recognition` contains:
- The dataset for the training of the model created using `make_dateset.py`;
- Different models trained for the hand gesture recognition, created using `train_model.py`:
  - `rino` containing notes about some machine learning models and how to use them;
  - `paolo` containing some scikit-learn models, included the final one used in the project (SVM);
  - `test_gesture_HMM.py` implementing a Hidden Markov Model Filtering for the gesture recognition;
- `test_camera.py` for testing camera and openCV;
- `recycle_bin` containing some old files;
- `images` containing some images used in this README;
- `virtual_board_filtered.py` for the implementation of the virtual whiteboard without any filtering et with basic gesture recognition;
- `virtual_board_filtered.py` for the implementation of the virtual whiteboard with HMM and Kalman filtering and trained model for gesture recognition;
- `demo_kalman.py` implementing a Kalman filter on finger position;
- `requirements.txt` for installing the required libraries with pip.

## Filtering

