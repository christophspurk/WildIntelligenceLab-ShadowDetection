import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


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

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')