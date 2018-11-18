# coding: utf-8

import os
import numpy as np
from keras import backend as K
import keras
import tensorflow as tf
import pandas as pd
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import cv2

SEED = 666
SHAPE = (512, 512, 4)
DIR = '/data/d14122793/human_protein_atlas_image_classification'
ia.seed(SEED)


# credits: https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302
# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class ProteinDataGenerator(keras.utils.Sequence):

    def __init__(self, paths, labels, batch_size, shape, shuffle=False, use_cache=False, augment=False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]

        if self.augment == True:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),  # horizontal flips
                    iaa.Crop(percent=(0, 0.1)),  # random crops
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5,
                                  iaa.GaussianBlur(sigma=(0, 0.5))
                                  ),
                    # Strengthen or weaken the contrast in each image.
                    iaa.ContrastNormalization((0.75, 1.5)),
                    # Add gaussian noise.
                    # For 50% of all images, we sample the noise once per pixel.
                    # For the other 50% of all images, we sample the noise per pixel AND
                    # channel. This can change the color (not only brightness) of the
                    # pixels.
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])], random_order=True)

            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y, y), 0)

        return X, y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R),
            np.array(G),
            np.array(B),
            np.array(Y)), -1)

        im = cv2.resize(im, (SHAPE[0], SHAPE[1]))
        im = np.divide(im, 255)
        return im

    def getTrainDataset(self):
        path_to_train = DIR + '/train/'
        data = pd.read_csv(DIR + '/train.csv')
        paths = []
        labels = []

        for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
            y = np.zeros(28)
            for key in lbl:
                y[int(key)] = 1
            paths.append(os.path.join(path_to_train, name))
            labels.append(y)

        return np.array(paths), np.array(labels)

    def getTestDataset(self):
        path_to_test = DIR + '/test/'
        data = pd.read_csv(DIR + '/sample_submission.csv')
        paths = []
        labels = []

        for name in data['Id']:
            y = np.ones(28)
            paths.append(os.path.join(path_to_test, name))
            labels.append(y)

        return np.array(paths), np.array(labels)

    # credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
    def f1(self, y_true, y_pred):
        # y_pred = K.round(y_pred)
        # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2 * p * r / (p + r + K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)

    def f1_loss(self, y_true, y_pred):
        # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2 * p * r / (p + r + K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return 1 - K.mean(f1)
