import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from data_tf import DataGenerator
import os


# create train datagen

data_gen = DataGenerator()

train_dataset_info = data_gen.get_data()

train_datagen = data_gen.create_train(train_dataset_info, 5, (299, 299, 3))

images, labels = next(train_datagen)


# Build model and train

def create_model(input_shape, n_out):
    pretrain_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)

    model = Sequential()
    model.add(pretrain_model)
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_out))
    model.add(Activation('sigmoid'))
    return model


model = create_model(
    input_shape=(299,299,3),
    n_out=28)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-04),
    metrics=['acc'])

model.summary()

epochs = 1000
batch_size = 32
checkpointer = ModelCheckpoint(
    '/data/d14122793/human_protein_atlas_image_classification/InceptionResNetV2_tf.model',
    verbose=2,
    save_best_only=True)

early_stopper = EarlyStopping(patience=10)

# split and suffle data
np.random.seed(2018)
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes = indexes[:27500]
valid_indexes = indexes[27500:]

# create train and valid datagens
train_generator = data_gen.create_train(
    train_dataset_info[train_indexes], batch_size, (299, 299, 3))
validation_generator = data_gen.create_train(
    train_dataset_info[valid_indexes], 100, (299, 299, 3), augument=False)

# train model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=epochs,
    verbose=1,
    callbacks=[checkpointer, early_stopper])
