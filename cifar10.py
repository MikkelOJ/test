# Import libs
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, Flatten, MaxPool2D, Dropout
import keras
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


# Support functions
# -------------------------------------------------------------------------- #
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def showImage(imageId, ):
    rgbArray = cifarDict[b'data'][imageId, :]
    img_reshaped = np.transpose(np.reshape(rgbArray, (3, 32, 32)), (1, 2, 0))
    classId = cifarDict[b'labels'][imageId]
    label = cifarMetaDict[b'label_names'][classId].decode("utf-8")

    plt.imshow(img_reshaped)
    xpos = 1
    ypos = 1
    plt.text(xpos, ypos, label, bbox=dict(facecolor='white', alpha=0.5))


def showImageRec(rgbArray, channelMeans):
    rgbArray *= 255

    for i in np.arange(0, np.shape(rgbArray)[2]):
        rgbArray[:,:,i] += channelMeans[i]

    img_reshaped = np.transpose(np.reshape(rgbArray, (3, 32, 32)), (1, 2, 0))
    plt.imshow(img_reshaped)

# Setup parameters
# -------------------------------------------------------------------------- #
filename = 'data_batch_1'
filelocation = 'cifar-10-batches-py/'
fileId = filelocation + filename
verbose = False

# Import files
cifarDict = unpickle(fileId)
cifarMetaDict = unpickle(filelocation + 'batches.meta')


# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024
# entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored
# in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the
# image.

# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in
# the array data.

# The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the
# following entries: label_names -- a 10-element list which gives meaningful names to the numeric labels in the
# labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.

# Explore data and format
# -------------------------------------------------------------------------- #
x = cifarDict[b'data']
y = cifarDict[b'labels']
print("The data has the shape", np.shape(x))

nChannels = 3
nPixels = int(np.shape(x)[1] / nChannels)
imgHeight = int(np.sqrt(nPixels))
imgWidth = int(np.sqrt(nPixels))


# Find mean values
channelMeans = []
for iChan in range(0, nChannels):
    channelMeans.append(np.mean(x[:, iChan * nPixels: (iChan + 1) * nPixels - 1]))

binSize = 40
rVals = x[:,0:1023].flatten()
gVals = x[:,1024:2047].flatten()
bVals = x[:,2048:].flatten()

plt.subplot(1,3,1)
plt.hist(rVals,bins = binSize,alpha=0.5,color = 'red',label='Red')
plt.subplot(1,3,2)
plt.hist(gVals,bins = binSize,alpha=0.5,color = 'green',label='Green')
plt.subplot(1,3,3)
plt.hist(bVals,bins = binSize,alpha=0.5,color = 'blue',label='Blue')

# Zero centering
rValsCentered = x[:,0:1024] - channelMeans[0]
gValsCentered = x[:,1024:2048] - channelMeans[1]
bValsCentered = x[:,2048:] - channelMeans[2]

xCentered = np.concatenate((rValsCentered,gValsCentered,bValsCentered),axis=1)


if verbose:
    rgbArray = xCentered[testInd, :]
    testInd = 1
    plt.subplot(1,2,1)
    showImage(testInd)

    plt.subplot(1,2,2)
    img_reshaped = np.transpose(np.reshape(xCentered[testInd,:], (3, 32, 32)), (1, 2, 0))
    plt.imshow(img_reshaped)

# Scaling
scaleToMax = 1
maxBeforeStd = np.max(xCentered)
xScaled = xCentered * (scaleToMax/maxBeforeStd)

# Check that data is not sorted
if verbose:
    plt.figure()
    plt.plot(y[:100],'.')

# SANDBOX
# -------------------------------------------------------------------------- #

# Model parameters
numClasses = 10
modelLoss = "categorical_entrophy"

# Split data
x_train, x_test, y_train, y_test = train_test_split(xScaled, y, test_size=0.25, random_state=42)

# Since its categorical labels, we make y into a categorical matrix
y_train = keras.utils.to_categorical(y_train, numClasses)
y_test = keras.utils.to_categorical(y_test, numClasses)

# Reshape X into correct spacial format
x_train = x_train.reshape((-1, nChannels, imgHeight,imgWidth))
x_test = x_test.reshape((-1, nChannels, imgHeight,imgWidth))

# Switch so that values are formatted as "channel_last" (now it's channel first)
x_train = x_train.transpose((0,2,3,1))
x_test = x_test.transpose((0,2,3,1))


# Define model
model = Sequential()

# Add two layers with 32 filters
model.add(Conv2D(
    filters = 32, # Number of filters
    kernel_size = (3,3),  # Filter size that moves
    padding='same', # Padding ensure that activation map gets same size as input (image)
    activation='relu', # Activation function
    input_shape = (imgHeight,imgWidth,nChannels), #Shape of each image
    data_format = "channels_last", # How is data formatted?
    strides = (1,1))) # Step size

model.add(Conv2D(
    filters = 32, # Number of filters
    kernel_size = (3,3),  # Filter size that moves
    padding='same', # Padding ensure that activation map gets same size as input (image)
    activation='relu', # Activation function
    data_format = "channels_last", # How is data formatted?
    strides = (1,1))) # Step size


model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(
    filters = 64, # Number of filters
    kernel_size = (3,3),  # Filter size that moves
    padding='same', # Padding ensure that activation map gets same size as input (image)
    activation='relu', # Activation function
    data_format = "channels_last", # How is data formatted?
    strides = (1,1))) # Step size


model.add(Conv2D(
    filters = 64, # Number of filters
    kernel_size = (3,3),  # Filter size that moves
    padding='same', # Padding ensure that activation map gets same size as input (image)
    activation='relu', # Activation function
    data_format = "channels_last", # How is data formatted?
    strides = (1,1))) # Step size

model.add(MaxPool2D(pool_size=(2,2)))

# model.add(Conv2D(
#     filters = 192, # Number of filters
#     kernel_size = (3,3),  # Filter size that moves
#     padding='same', # Padding ensure that activation map gets same size as input (image)
#     activation='relu', # Activation function
#     data_format = "channels_last", # How is data formatted?
#     strides = (1,1))) # Step size
#
# model.add(Conv2D(
#     filters = 192, # Number of filters
#     kernel_size = (1,1),  # Filter size that moves
#     padding='same', # Padding ensure that activation map gets same size as input (image)
#     activation='relu', # Activation function
#     data_format = "channels_last", # How is data formatted?
#     strides = (1,1))) # Step size
#
# model.add(Conv2D(
#     filters = 10, # Number of filters
#     kernel_size = (1,1),  # Filter size that moves
#     padding='same', # Padding ensure that activation map gets same size as input (image)
#     activation='relu', # Activation function
#     data_format = "channels_last", # How is data formatted?
#     strides = (1,1))) # Step size
#


model.add(Flatten())
#
# model.add(Dropout(0.3))
#
#model.add(Dense(256, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.99)

# Compile the model
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, validation_split=0.33, epochs=5)

# Evaluate performance
model.summary()


plt.plot(history.history['acc'],label='Train')
plt.plot(history.history['val_acc'],label='Validation')
#plt.plot(history.history['loss'], label='loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best')

model.evaluate(x_test, y_test, verbose=1)
p = model.predict(x_test)


if False:
    from sklearn.metrics import confusion_matrix
    y_true = np.argmax(y_test,axis=1)
    y_pred = np.argmax(p,axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm)
    plt.colorbar()

if False:
    # Save or load model
    from keras.models import load_model

    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    model = load_model('my_model.h5')

if False:
    from keras.utils import plot_model

    plot_model(model, to_file='model.png')


# See https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

# Tune hyperparameters, set to random search
# Use batch normalization
# Coarse to fine search
# Adam for gradient decent beta1 = 0.9, beta2 = 0.99
# Regularization - (inverted maybe) dropout, param = 0.5 is common (fully connected layers)
# Create more data: flipping, cropping, stretching, distort colors of images

# We use three main types of layers to build ConvNet architectures: Convolutional Layer,
# Pooling Layer, and Fully-Connected Layer
#[INPUT - CONV - RELU - POOL - FC]

