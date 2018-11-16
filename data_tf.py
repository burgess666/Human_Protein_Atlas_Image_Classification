import numpy as np
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa


class DataGenerator:

    def get_data(self):
        data_folder = '/data/d14122793/human_protein_atlas_image_classification'
        path_to_train = os.path.join(data_folder, 'train')
        data = pd.read_csv(os.path.join(data_folder, 'train.csv'))

        train_dataset_info = []
        for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
            train_dataset_info.append({
                'path': os.path.join(path_to_train, name),
                'labels': np.array([int(label) for label in labels])})
        train_dataset_info = np.array(train_dataset_info)

        return train_dataset_info

    def create_train(self, dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = self.load_image(
                    dataset_info[idx]['path'], shape)
                if augument:
                    image = self.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels

    @staticmethod
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

    @staticmethod
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
