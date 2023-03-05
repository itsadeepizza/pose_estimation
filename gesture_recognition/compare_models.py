import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


df_test = pd.concat([pd.read_csv('dataset/'+labels + '.csv') for labels in ['hand_opened', 'hand_closed', 'one', 'spiderman']])
df_train = pd.concat([pd.read_csv('dataset/'+labels + '_v2.csv') for labels in ['hand_opened', 'hand_closed', 'one', 'spiderman']])
for df in [df_test, df_train]:
    df['is_right'] = 0
    df.loc[df['handedness'] == 'Right', 'is_right'] = 1
    df.drop(columns=['handedness'], inplace=True)
# replace 'gesture' column by multiple dummy columns
labels_train = pd.get_dummies(df_train['gesture'])
df_train.drop(columns=['gesture'], inplace=True)
labels_test = pd.get_dummies(df_test['gesture'])
df_test.drop(columns=['gesture'], inplace=True)

# USe a random forest classifier
from sklearn.ensemble import RandomForestClassifier

# Convert dummy variable to integer
train_y = labels_train.values.argmax(axis=1)
test_y = labels_test.values.argmax(axis=1)
# Convert dataframes to numpy arrays
train_X = df_train.values
test_X = df_test.values





#For each position, calculate the mean of all landmarks, and subtract it from the landmarks
#This will make the model invariant to the position of the hand

for features in [train_X, test_X]:
    x_mean = features[:,0:-1:3].mean(axis=1)
    y_mean = features[:,1:-1:3].mean(axis=1)
    z_mean = features[:,2:-1:3].mean(axis=1)

    features[:,0:-1:3] -= x_mean[:,None]
    features[:,1:-1:3] -= y_mean[:,None]
    features[:,2:-1:3] -= z_mean[:,None]

scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

#
#
# # Make a grid search to find the best parameters
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
#     'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# }
# # Create a based model
# rf = RandomForestClassifier()
#
#
# # Test the model with the best parameters
# clf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], random_state=0)
#
#
# clf = RandomForestClassifier(n_estimators=600, max_depth=10, random_state=0)
# 0.8784615384615385
# [[ 390    6   66    6]
#  [   4  443    9    8]
#  [ 108   23 1043   29]
#  [  12   30   15  408]]

# Make a svm classifier

clf = SVC(C=10, gamma=0.001, kernel='rbf', random_state=0)

# 0.9023076923076923
# [[ 398   10   52    8]
#  [   2  456    3    3]
#  [  87    5 1086   25]
#  [  10   42    7  406]]

# Make a KNN classifier

# clf = KNeighborsClassifier(n_neighbors=5)
#
# # 0.8765384615384615
# # [[ 403    6   55    4]
# #  [   2  455    6    1]
# #  [ 130   26 1007   40]
# #  [  11   27   13  414]]
#
#
# param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]}
#
# # Grid search on KNN
#
#
# # param_grid = { 'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto'], 'kernel': ['rbf'], 'random_state': [0]}
# #
# # Instantiate the grid search model
# grid_search = GridSearchCV(param_grid=param_grid, estimator=clf, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)
# # Fit the grid search to the data with parallel processing
# grid_search.fit(train_X, train_y)
#
# # Print the results as a table
# results = pd.DataFrame(grid_search.cv_results_)
# print(grid_search.best_params_)
# print(results)
#
# clf = KNeighborsClassifier(**grid_search.best_params_)




clf.fit(train_X, train_y)
classes = labels_train.columns
print(classes)
# Test the model
from sklearn.metrics import accuracy_score
pred_y = clf.predict(test_X)
print(accuracy_score(test_y, pred_y))
# Print confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, pred_y))

