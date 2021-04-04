import h5py
import tensorflow as tf
import os

    """
    Returns:
    updated dataframe with added classification labels and confidence scores
    """

def classify_shadows(rgb_shadows, df_shadows, weights_path):
    # load pretrained model with weights
    # os.chdir(weights_path)
    model = tf.keras.applications.DenseNet121(include_top=True, classes=3, pooling=None)
    model.load_weights(weights_path)
    # predict rgb_cuts with loaded model
    classes = []
    confidence_scores = []
    predicted_label = []
    predictions = model.predict(rgb_shadows)
    for p in predictions:
        predicted_label.append(np.argmax(predictions[p]))

    # score = tf.nn.softmax(predictions[0])
    # classes.append(class_names[np.argmax(score)])
    # confidence_scores.append(round(100 * np.max(score), 3))
    df_shadows['class'] = classes
    df_shadows['confidence_score'] = confidence_scores

    return df_shadows