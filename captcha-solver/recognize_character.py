import os
import json
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import optimizers, losses, metrics
from keras.preprocessing.image import ImageDataGenerator


def createModel(input_shape, nClasses):
    model = Sequential()
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
        
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))
    
    return model


train_data_dir = "character_extraction/train/generated_images"
validation_data_dir = "character_extraction/validation"
img_width = 21
img_height = 38
input_shape = (img_height, img_width, 1,)
nClasses = 19

# Hyperparameters
batch_size = 10
epochs = 100

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    color_mode = "grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)


validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    color_mode = "grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)


model = createModel(input_shape, nClasses)
model.summary()


optimizer = optimizers.SGD(lr=0.0001)
loss = 'categorical_crossentropy'
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = model.fit_generator(train_generator,
	epochs=epochs, verbose=1, use_multiprocessing=True, 
    shuffle=True, validation_data=validation_generator,
    steps_per_epoch=len(train_generator), validation_steps=len(validation_generator))

with open('trainHistoryDict', 'w+') as file_pi:
        json.dump(history.history, file_pi)

model.save('model.h5')