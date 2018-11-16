import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import metrics
from keras.optimizers import Adam
from data_tf import DataGenerator
import os
import time


# create train datagen

data_gen = DataGenerator()

train_dataset_info = data_gen.get_data()

train_datagen = data_gen.create_train(train_dataset_info, 5, (299, 299, 3))

images, labels = next(train_datagen)


# Build model and train

def create_model(input_shape, n_out):
    pretrain_model = ResNet50(
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


model = create_model(input_shape=(299, 299, 3), n_out=28)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-04),
    metrics=['acc'])

model.summary()

epochs = 1000
batch_size = 16

checkpointer = ModelCheckpoint(
    filepath=os.path.join('//data/d14122793/human_protein_atlas_image_classification', 'checkpoints', 'Resnet-Training-' + \
                          '{epoch:03d}-{val_loss:.3f}.hdf5'),
    verbose=1,
    save_best_only=True)

early_stopper = EarlyStopping(patience=50)

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('.', 'logs', 'Resnet50-Training-' + str(timestamp) + '.log'))


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
    steps_per_epoch=1718,
    validation_data=next(validation_generator),
    epochs=epochs,
    verbose=1,
    callbacks=[checkpointer, early_stopper, csv_logger])
