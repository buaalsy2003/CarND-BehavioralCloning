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

imgSizeX = 80
imgSizeY = 300
smallSizeX = 20
smallSizeY = 75

#Remove the first row because it is column header
data = data[1:]
angles = ()
length = len(data)
center = np.ndarray(shape=(length * 2, smallSizeX, smallSizeY, 3))
#center = np.ndarray(shape=(1000 * 2, smallSizeX, smallSizeY, 3))

for i in range(length):
    angle = float(data[i][3])
    angles += (angle, )
    img = ndimage.imread(data[i][0]).astype(np.float32)[60:-20, 10:-10, :]
    img = imresize(img, (smallSizeX, smallSizeY, 3))
    center[i*2] = img
    if(angle > 0.0):
        angles += (-float(data[i][3]), )
    else:
        angles += (0.0, )
    center[i*2+1] = np.fliplr(img)
    
angles = np.array(angles)

#print (center[0].shape)
#plt.imshow(center[19])
#plt.imshow(center[1])
#plt.show()
#print("data size:", angles[18])
#print("data size:", angles[19])

print (center.shape)
print (angles.shape)


center, angles = shuffle(center, angles)

# Get randomized datasets for training and test
X_train, X_test, y_train, y_test = train_test_split(
    center,
    angles,
    test_size=0.05,
    random_state=2017)

# Get randomized datasets for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.05,
    random_state=2017)
    
X_train = X_train / 255 - 0.5
X_valid = X_valid / 255 - 0.5
X_test  = X_test  / 255 - 0.5
    
print (X_train.shape)
print (X_test.shape)
print (X_valid.shape)

#architecture
model = Sequential()
model.add(BatchNormalization(axis=1, input_shape=(20,75,3)))
model.add(Conv2D(24, 3, 3, border_mode='valid', subsample=(2,2), activation='relu')) 
model.add(Conv2D(36, 3, 3, border_mode='valid', subsample=(1,2), activation='relu'))
model.add(Conv2D(48, 3, 3, border_mode='valid', activation='relu'))
model.add(Conv2D(64, 2, 2, border_mode='valid', activation='relu'))
model.add(Conv2D(64, 2, 2, border_mode='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(64))
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
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

json_string = model.to_json()
with open('model.json', 'w') as jsonfile:
    json.dump(json_string, jsonfile)
    model.save_weights('model.h5')
print("Model Saved")
                    
                    
