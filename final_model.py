from morph import get_shadows
from classify_shadows import classify_shadows
from visualization import classification_visualization
from knn_model import get_classifier, get_binary

import tensorflow as tf
import matplotlib.pyplot as plt

class MODELS:

    def __init__(self, filepath_knn, filepath_class):

        #### run automatically when the object is created: ###
        self.knn_model = get_classifier(filepath_knn, 3)
        self.classification_model = self.load_classification_model(filepath_class)


    def load_classification_model(self, filepath_class):
        # tf.keras.models.load_model(self.filepath_class, custom_objects=None, compile=True, options=None)
        # The operation of classification/ image prediciton would be defined within this method
        model = tf.keras.applications.DenseNet121(include_top=True,
                                                  classes=3, weights=None, pooling=None)
        model.load_weights(filepath_class)
        return model # classification result

    def model_predict(self, rgb_img, img_name, trans_matrix):
        bin_img = get_binary(self.knn_model, rgb_img, 256)

        bin_cuts, rgb_cuts, df_morph = get_shadows(bin_img, rgb_img,
                                                   img_name, trans_matrix)
        df_morph = classify_shadows(rgb_cuts, df_morph, self.classification_model)
        rgb_labeled_img = classification_visualization(df_morph, bin_img, rgb_img)

        return rgb_labeled_img, df_morph

