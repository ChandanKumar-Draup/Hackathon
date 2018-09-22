##### CNN model for Object Recognition in Photographs of CIFAR_10 dataset (colored images 32 * 32 size )
import numpy
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#background Theano library in Keras
K.set_image_dim_ordering('th')

#random seed is fixed for reproducibility
seed = 7
numpy.random.seed(seed)

#loaded entire cifar dataset, datasets are pickled whch is converted using cifar10.load_data function in Keras library
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to -1.0-1.0
X_train = X_train[0:800].astype('float32')
X_test = X_test[800:1000].astype('float32')
X_train = (X_train -128)/ 128.0
X_test = (X_test -128) / 128.0

# one hot encoding of outputs, to assign individual integers to each of 10 class from 0..9
y_train = np_utils.to_categorical(y_train[0:800])
y_test = np_utils.to_categorical(y_test[800:1000])
num_classes = y_test.shape[1]

#data augmentation, to preprocess data like rotation, image location
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(X_train)

# Createion of the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
#batch_size = 32
batch_size=64
epochs = 150
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),verbose=1,validation_data=(X_test,y_test),steps_per_epoch=X_train.shape[0] / batch_size,epochs= epochs)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
