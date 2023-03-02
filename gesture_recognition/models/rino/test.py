import pandas as pd
#import time

features = pd.concat([pd.read_csv('../../dataset_v1/'+labels + '.csv') for labels in ['hand_opened', 'hand_closed', 'one', 'spiderman']])
featuresv2 = pd.concat([pd.read_csv('../../dataset_v2/'+labels + '_v2.csv') for labels in ['hand_opened', 'hand_closed', 'one', 'spiderman']])
dataset = pd.concat([features,featuresv2])

dataset['is_right'] = 0
dataset.loc[dataset['handedness'] == 'Right', 'is_right'] = 1
dataset.drop(columns=['handedness'], inplace=True)


# crea una matrice hone hot encoding dalla colonna gesture
labels = pd.get_dummies(dataset['gesture'])

dataset.drop(columns=['gesture'], inplace=True)
#dataset=pd.concat([dataset,labels],axis=1)

#dataset:
#z0	x1	y1	z1	x2	y2	z2	x3	...	z19	x20	y20	z20

#labels:
# 'gesture',
# 'is_right',
# 'hand_closed',
# 'hand_opened',
# 'one',
# 'spiderman'],
#