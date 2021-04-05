import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import cv2

def get_classifier(name_path,number_neighbours):
    dataset = pd.read_csv(name_path,index_col=0)
  # delete unecessary columns
    dataset = dataset.drop(columns=['Unnamed: 5', 'Unnamed: 6'])

  # # creating labelEncoder
    le = preprocessing.LabelEncoder()
    dataset['encoded'] = le.fit_transform(dataset.label)

  # Separate majority and minority classe

    dataset_majority = dataset[dataset.encoded==0]
    dataset_minority = dataset[dataset.encoded==1]
    # upsample minority class

    dataset_minortiy_upsampled = resample(dataset_minority,
                                        replace=True,
                                        n_samples = dataset.groupby('label')['encoded'].count().max(), #to match majority class
                                        random_state=5)  #reproducible resul

    # combine the majority and minortiy dataset
    dataset_combine = pd.concat([dataset_majority,dataset_minortiy_upsampled])

    #split data set into features and label
    X = dataset_combine.iloc[:,:-2].values
    y = dataset_combine.iloc[:,4].values

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.05) # 80% training and 20% test

    #classification with KNN (n = 3)

    classifier = KNeighborsClassifier(n_neighbors=int(number_neighbours))
    classifier.fit(X_train, y_train)

    return classifier


def get_binary(classifier, img, resize):
    img = cv2.resize(img, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)

    # reshape img array
    arr = np.array(img)
    shape = arr.shape
    flat_arr = arr.reshape(shape[0] * shape[1], 3)
    vector = np.matrix(flat_arr)
    arr2 = np.asarray(vector).reshape(shape)

    # put the image into KNN Model as a test dataset
    y_pred_img = classifier.predict(flat_arr)

    # reshape y_pred -> 2048
    y_predI = (np.reshape(y_pred_img, (resize, resize))).astype(np.uint8)
    y_predII = cv2.resize(y_predI, dsize=(2048, 2048))

    return y_predII
