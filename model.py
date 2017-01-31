import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import json
import h5py

### Import data
data = []
with open('driving_log.csv') as F:
    reader = csv.reader(F)
    for i in reader:
        data.append(i) 

#imgSizeX = 100
#imgSizeY = 320
smallSizeX = 20
smallSizeY = 75

#Remove the first row because it is column header
data = data[1:]
length = len(data)
angles = ()
otherangles = ()
center = []
other = []

for i in range(length):
    angle = float(data[i][3])
    angles += (angle, )
    img = ndimage.imread(data[i][0].strip()).astype(np.float32)[60:-20, 10:-10, :]
    img = imresize(img, (smallSizeX, smallSizeY, 3))
    center.append(img)	
	#flip the center img if the anlge is not zero
    if(angle > 0.0 or angle < 0.0 ):
        otherangles += (-float(data[i][3]), )
        other.append(np.fliplr(img))
	
	    #left
        img = ndimage.imread(data[i][1].strip()).astype(np.float32)[60:-20, 10:-10, :]
        img = imresize(img, (smallSizeX, smallSizeY, 3))
        other.append(img)
        otherangles += (angle + 0.03, )
	
	    #right
        img = ndimage.imread(data[i][2].strip()).astype(np.float32)[60:-20, 10:-10, :]
        img = imresize(img, (smallSizeX, smallSizeY, 3))
        other.append(img)
        otherangles += (angle - 0.03, )

	
labels = np.array(angles)
features = np.array(center)

otherLabels = np.array(otherangles)
otherFeatures = np.array(other)

#print (center[0].shape)
#plt.imshow(other[19])
#plt.imshow(center[1])
plt.show()
print("Angle:", otherangles[19])
#print("Angle:", angles[1])

print (features.shape)
print (labels.shape)


features, labels = shuffle(features, labels)
otherFeatures, otherLabels = shuffle(otherFeatures, otherLabels)

# Get randomized datasets for training and test
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.1,
    random_state=2017)

# Get randomized datasets for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=2017)

#only append the left and right images into training datasets
X_train = np.concatenate((X_train, otherFeatures), axis = 0)
y_train = np.concatenate((y_train, otherLabels), axis = 0)

X_train, y_train = shuffle(X_train, y_train)
    
X_train = X_train / 255 - 0.5
X_valid = X_valid / 255 - 0.5
X_test  = X_test  / 255 - 0.5
    
print (X_train.shape)
print (X_test.shape)
print (X_valid.shape)

#architecture
model = Sequential()
model.add(Conv2D(24, 3, 3, border_mode='valid', subsample=(2,2), activation='relu', input_shape=(smallSizeX,smallSizeY,3))) 
model.add(Conv2D(36, 3, 3, border_mode='valid', subsample=(1,2), activation='relu'))
model.add(Conv2D(48, 3, 3, border_mode='valid', activation='relu'))
model.add(Conv2D(64, 2, 2, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.summary()


# Compile model with adam optimizer and learning rate of .0001
# and loss computed by mean squared error
#opt = SGD(lr=0.01)
opt = Adam(lr = 0.0001)
model.compile(loss='mean_squared_error',
              optimizer= opt,
              metrics=['accuracy'])

### Model training
history = model.fit(X_train, y_train,
                    batch_size=128, nb_epoch=10,
                    verbose=1, shuffle=True, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

json_string = model.to_json()
with open('model.json', 'w') as jsonfile:
    json.dump(json_string, jsonfile)
    model.save_weights('model.h5')
print("Model Saved")