import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from gesture_recognition.config import selected_config as conf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class BaseGestureRecognitionModel():

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.param_grid = None


    def prepare_dataset(self):
        df_test = pd.concat([pd.read_csv(conf.PATH_DATASET +'/' + label + '.csv') for label in ['hand_opened', 'hand_closed', 'one', 'spiderman']])
        df_train = pd.concat([pd.read_csv(conf.PATH_DATASET +'/' + label + '_v2.csv') for label in ['hand_opened', 'hand_closed', 'one', 'spiderman']])
        for df in [df_test, df_train]:
            df['is_right'] = 0
            df.loc[df['handedness'] == 'Right', 'is_right'] = 1
            df.drop(columns=['handedness'], inplace=True)
        # Keep only left hands (as the dataset contains only left hands
        df = df[df['is_right'] == 0]
        df.drop(columns=['is_right'], inplace=True)

        # replace 'gesture' column by multiple dummy columns
        labels_train = pd.get_dummies(df_train['gesture'])
        df_train.drop(columns=['gesture'], inplace=True)
        labels_test = pd.get_dummies(df_test['gesture'])
        df_test.drop(columns=['gesture'], inplace=True)
        # Convert dummy variable to integer
        train_y = labels_train.values.argmax(axis=1)
        test_y = labels_test.values.argmax(axis=1)
        # Convert dataframes to numpy arrays
        train_X = df_train.values
        test_X = df_test.values
        self.classes = labels_train.columns

        return train_X, train_y, test_X, test_y

        # Normalize mean and variance for each feature

    def fit(self):
        train_X, train_y, test_X, test_y = self.prepare_dataset()
        # For each position, calculate the mean of all landmarks, and subtract it from the landmarks
        # This will make the model invariant to the position of the hand
        train_X = self.normalise(train_X)
        test_X = self.normalise(test_X)

        self.scaler.fit(train_X)
        train_X = self.scaler.transform(train_X)
        test_X = self.scaler.transform(test_X)

        self.model.fit(train_X, train_y)
        #Test the model
        pred_y = self.model.predict(test_X)
        print(accuracy_score(test_y, pred_y))
        print(self.classes)
        print("Row: true label, column: predicted label")
        print(confusion_matrix(test_y, pred_y, normalize='true'))
        # Show confusion matrix as a heatmap
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.heatmap(confusion_matrix(test_y, pred_y, normalize='true'), annot=True, xticklabels=self.classes, yticklabels=self.classes)
        plt.show()


    def predict_proba(self, hand_landmarks):
        raise NotImplementedError

    def save(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(
                {'model': self.model,
                 'scaler': self.scaler,
                 'classes': self.classes,
                }, f)

    def load(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.classes = data['classes']

    def normalise(self, X):
        # For each position, calculate the mean of all landmarks, and subtract it from the landmarks
        # This will make the model invariant to the position of the hand
        n = X.shape[0]
        X_as_vectors = np.array([X[:, 0:-1:3], X[:, 1:-1:3], X[:, 2:-1:3]])
        # rotate axis
        X_as_vectors = np.moveaxis(X_as_vectors, 0, -1)
        # Center on the origin
        X_as_vectors -= X_as_vectors.mean(axis=1)[:, None, :]


        # Calculate vector landmark 0 -> landmark 5
        # This will make the model invariant to the orientation of the hand
        hand_vector = X_as_vectors[:, 5, :] - X_as_vectors[:, 0, :]
        # normalise
        hand_vector /= np.linalg.norm(hand_vector, axis=1)[:, None]
        # Calculate the angle between the vector projection on the x-y plane and the x axis
        angle_x = np.arctan2(hand_vector[:, 1], hand_vector[:, 0])

        # Calculate the 3D rotation matrix for aligning the projection of the hand vector with the x axis
        rotation_matrix_xy = np.array([
            [np.cos(angle_x), -np.sin(angle_x), np.zeros(n)],
            [np.sin(angle_x), np.cos(angle_x), np.zeros(n)],
            [np.zeros(n), np.zeros(n), np.ones(n)]
        ])
        # Rotate the landmarks
        for i in range(n):
            vecs = X_as_vectors[i, :, :]
            rot_mat = rotation_matrix_xy[:, :, i]
            X_as_vectors[i, :, :] = np.matmul(vecs, rot_mat)

        # Update hand vector
        hand_vector = X_as_vectors[:, 5, :] - X_as_vectors[:, 0, :]
        # normalise
        hand_vector /= np.linalg.norm(hand_vector, axis=1)[:, None]
        # Calculate the angle between the hand vector and the z axis
        angle_z = np.arctan2(hand_vector[:, 2], hand_vector[:, 0])
        # Calculate the 3D rotation matrix for aligning the hand vector with the x axis
        rotation_matrix_xz = np.array([
            [np.cos(angle_z), np.zeros(n), -np.sin(angle_z)],
            [np.zeros(n), np.ones(n), np.zeros(n)],
            [np.sin(angle_z), np.zeros(n), np.cos(angle_z)]
        ])
        # Rotate the landmarks
        for i in range(n):
            vecs = X_as_vectors[i, :, :]
            rot_mat = rotation_matrix_xz[:, :, i]
            X_as_vectors[i, :, :] = np.matmul(vecs, rot_mat)
        # Now the and vector is aligned with the x axis

        # COnvert back to 1D array
        X_normalised = np.reshape(X_as_vectors, (X_as_vectors.shape[0], -1))
        return X_normalised

    def grid_search(self):
        train_X, train_y, _, _ = self.prepare_dataset()
        # For each position, calculate the mean of all landmarks, and subtract it from the landmarks
        # This will make the model invariant to the position of the hand
        self.scaler.fit(train_X)
        train_X = self.normalise(train_X)

        # Instantiate the grid search model
        grid_search = GridSearchCV(param_grid=self.param_grid, estimator=self.model, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)
        # Fit the grid search to the data with parallel processing
        grid_search.fit(train_X, train_y)
        # Print results for each parameter combination

        print(grid_search.best_params_)
        results = pd.DataFrame(grid_search.cv_results_)
        print(results[['params', 'mean_test_score', 'rank_test_score']])

    def normalise_landmarks(self, hand_landmarks, handeness):
        landmark_list = []
        invert_hand = 1 if handeness == "Left" else -1
        for landmark_idx, coords in enumerate(hand_landmarks.landmark):
            landmark_list.append(coords.x * invert_hand)
            landmark_list.append(coords.y)
            landmark_list.append(coords.z)
        # We predict using left hand, but invert x coordinates if the hand is right
        # Pay attention, handeness is inverted in the dataset
        landmark_list.append(0)
        X = np.array([landmark_list])
        X = self.normalise(X)
        X = self.scaler.transform(X)
        return X


class SVCModel(BaseGestureRecognitionModel):
    def __init__(self):
        super().__init__()
        self.model = SVC(C=10, gamma=0.001, kernel='rbf', random_state=0, probability=True)
        self.param_grid = { 'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto'], 'kernel': ['rbf'], 'random_state': [0]}

    def predict_proba(self, hand_landmarks, handeness):
        landmark_list = []
        for landmark_idx, coords in enumerate(hand_landmarks.landmark):
            landmark_list.append(coords.x)
            landmark_list.append(coords.y)
            landmark_list.append(coords.z)
        landmark_list.append(1 if handeness == "Right" else 0)
        X = np.array([landmark_list])
        X = self.normalise(X)
        X = self.scaler.transform(X)
        probs = self.model.predict_proba(X)
        return {gesture: prob for gesture, prob in zip(self.classes, probs[0])}

if __name__ == '__main__':
    model = SVCModel()
    model.fit()
    model.save('svc.pkl')
