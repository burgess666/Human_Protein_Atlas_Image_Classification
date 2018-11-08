import csv
import numpy as np
import random
import glob
import os.path
import pandas as pd
import sys
import operator
import threading
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm


'''
class Dataset:

    def __init__(self):
        # Folder path on server
        # self.data_folder = '/data/d14122793/human_protein_atlas_image_classification'

        # Folder Path on Mac
        self.data_folder = '/Users/Kellan/Desktop/Human_Protein_Atlas_Image_Classification'

        # Get the data.
        self.data = self.get_data()

    def get_data(self):
        # Load  data from file
        with open(os.path.join(self.data_folder, 'train.csv'), 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            data = list(reader)

        return data

    def get_classes(self):
        # Get classes
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        return classes


if __name__ == '__main__':
    data = Dataset()
    print(data.get_classes())
'''

data_folder = '/data/d14122793/human_protein_atlas_image_classification'
#data_folder = '/Users/Kellan/Desktop/Human_Protein_Atlas_Image_Classification'
path_to_train = os.path.join(data_folder, 'train')
data = pd.read_csv(os.path.join(data_folder, 'train.csv'))

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path': os.path.join(path_to_train, name),
        'labels': np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)


class data_generator:

    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels

    def load_image(path, shape):
        image_red_ch = skimage.io.imread(path + '_red.png')
        image_yellow_ch = skimage.io.imread(path + '_yellow.png')
        image_green_ch = skimage.io.imread(path + '_green.png')
        image_blue_ch = skimage.io.imread(path + '_blue.png')

        image_red_ch += (image_yellow_ch / 2).astype(np.uint8)
        image_green_ch += (image_yellow_ch / 2).astype(np.uint8)

        image = np.stack((
            image_red_ch,
            image_green_ch,
            image_blue_ch), -1)
        image = resize(image, (shape[0], shape[1]), mode='reflect')
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

# create train datagen
train_datagen = data_generator.create_train(
    train_dataset_info, 5, (299,299,3), augument=True)

images, labels = next(train_datagen)

fig, ax = plt.subplots(1,5,figsize=(25,5))
for i in range(5):
    ax[i].imshow(images[i])
print('min: {0}, max: {1}'.format(images.min(), images.max()))





from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import keras


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

keras.backend.clear_session()

model = create_model(
    input_shape=(299,299,3),
    n_out=28)

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-04),
    metrics=['acc'])

model.summary()

epochs = 100; batch_size = 16
checkpointer = ModelCheckpoint(
    '/Users/Kellan/Desktop/Human_Protein_Atlas_Image_Classification/InceptionResNetV2.model',
    verbose=2,
    save_best_only=True)

# split and suffle data
np.random.seed(2018)
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes = indexes[:27500]
valid_indexes = indexes[27500:]

# create train and valid datagens
train_generator = data_generator.create_train(
    train_dataset_info[train_indexes], batch_size, (299,299,3), augument=True)
validation_generator = data_generator.create_train(
    train_dataset_info[valid_indexes], 100, (299,299,3), augument=False)

# train model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=epochs,
    verbose=1,
    callbacks=[checkpointer])
