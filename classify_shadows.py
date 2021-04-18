import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

"""
    Returns:
    updated dataframe with added classification labels and confidence scores
"""
def classify_shadows(rgb_shadows, df_shadows, model):
    # load pretrained model with weights
    # os.chdir(weights_path)


    # predict rgb_cuts with loaded model
    classes = []
    confidence_scores = []

    for arr in rgb_shadows:
        # DenseNet only accepts shape (224,224,3)
        new_size = cv2.resize(arr, (224,224))
        label, score = get_prediction_score(new_size, model)
        classes.append(label)
        confidence_scores.append(score)

    df_shadows['class'] = classes
    df_shadows['confidence_score'] = confidence_scores

    return df_shadows

def get_prediction_score(img_array, model):
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # TODO: how to extract class names automatically
    class_names = ['animal', 'bush', 'tree']
    class_label = class_names[np.argmax(score)]

    confidence_score = round(100 * np.max(score), 3)

    return class_label, confidence_score
