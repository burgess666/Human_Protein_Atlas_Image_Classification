import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from data_tf import DataGenerator
from keras.models import load_model


data_gen = DataGenerator()
submit = pd.read_csv('/data/d14122793/human_protein_atlas_image_classification/sample_submission.csv')
checkpints_path = '/data/d14122793/human_protein_atlas_image_classification/checkpoints'
model = load_model(os.path.join(checkpints_path, 'Resnet-Training-083-3.195.hdf5'))

predicted = []

for name in tqdm(submit['Id']):
    path = os.path.join('/data/d14122793/human_protein_atlas_image_classification/test/', name)
    image = data_gen.load_image(path, (299, 299, 3))
    score_predict = model.predict(image[np.newaxis])[0]
    label_predict = np.arange(28)[score_predict>=0.5]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)


submit['Predicted'] = predicted
submit.to_csv('submission008.csv', index=False)
