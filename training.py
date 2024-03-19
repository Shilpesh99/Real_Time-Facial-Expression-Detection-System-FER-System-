import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os

num_face_labels = 5
img_rows,img_cols = 48,48
batch_size = 8

train_data_directory = r'C:\Users\Acer\OneDrive\Desktop\ML\FER system\train dataset'
validation_data_directory = r'C:\Users\Acer\OneDrive\Desktop\ML\FER system\validation dataset'

train_datagenerator = ImageDataGenerator(
                        rescale=1./255, 
                        rotation_range=30, 
                        shear_range=0.3, 
                        zoom_range=0.3, 
                        width_shift_range=0.4, 
                        height_shift_range=0.4, 
                        horizontal_flip=True,
                        fill_mode='nearest')

validation_datagenerator = ImageDataGenerator(rescale=1./255)

train_generator = train_datagenerator.flow_from_directory(
                        train_data_directory, 
                        color_mode='grayscale', 
                        target_size=(img_rows,img_cols), 
                        batch_size=batch_size, 
                        class_mode='categorical', 
                        shuffle=True
                        )

validation_generator = validation_datagenerator.flow_from_directory(
                        validation_data_directory,
                        color_mode='grayscale', 
                        target_size=(img_rows,img_cols), 
                        batch_size=batch_size, 
                        class_mode='categorical', 
                        shuffle=True
                        )

model = Sequential()

# Block 1
model.add(Conv2D(32,(3,3), padding="same", kernel_initializer="he_normal", input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3), padding="same", kernel_initializer="he_normal", input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block 2 
model.add(Conv2D(64,(3,3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block 3 
model.add(Conv2D(128,(3,3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block 4 
model.add(Conv2D(256,(3,3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block 5 
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block 6
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block 7
model.add(Dense(num_face_labels,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(r'C:\Users\Acer\OneDrive\Desktop\ML\FER system\Emotion_little_vgg.h5',
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True,
                            verbose=1
                            )

earlystop = EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=3,
                            verbose=1,
                            restore_best_weights=True
                            )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                            factor=0.2,
                            patience=3,
                            verbose=1,
                            min_delta=0.0001
                            )

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
            optimizer=adam_v2.Adam(lr=0.0001),
            metrics=['accuracy'])

nb_train_samples = 24282
nb_validation_samples = 5937
epochs = 25

history = model.fit_generator(
                            train_generator,
                            steps_per_epoch=nb_train_samples//batch_size,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=validation_generator,
                            validation_steps=nb_validation_samples//batch_size
                            )

