import pandas as pd
import time

features = pd.concat([pd.read_csv(labels + '.csv') for labels in ['hand_opened', 'hand_closed', 'one', 'spiderman']])
features['is_right'] = 0
features.loc[features['handedness'] == 'Right', 'is_right'] = 1
features.drop(columns=['handedness'], inplace=True)
# replace 'gesture' column by multiple dummy columns
labels = pd.get_dummies(features['gesture'])

features.drop(columns=['gesture'], inplace=True)

# USe a random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(train_X, train_y)

# Test the model
from sklearn.metrics import accuracy_score
pred_y = clf.predict(test_X)
print(accuracy_score(test_y, pred_y))
# Print confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, pred_y))

