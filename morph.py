# -*- coding: utf-8 -*-

import math
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

image_name = "test_dummy_bin_bla_2048_2048.png"
# kernels for morph transformation, if == 0 kernel will be ignored
KER_ER1 = np.ones((3, 3), np.uint8)  # for humid (3,3); for dry (2,2)
KER_DI1 = np.ones((30, 30), np.uint8)  # for humid (30,30); for dry (20,20)
KER_ER2 = np.ones((7, 7), np.uint8)
KER_DI2 = 0  # np.ones((1,1),np.uint8)
# add space on left(x) and top(y) of contour bounding box for cut out
X_ADD = 50
Y_ADD = 50
# gps_matrix - does not work for full folder, as the matrix changes depending on
# the coordinates of the mask_slice
GPS_TRANS = 0  # ortho.transform #if == 0, df_cnt['gps'] = 0
# min/max area for contours to be relevant
MIN_AREA = 4600  # smaller is often grass
MAX_AREA = 250000  # larger is often black area outside picture
LINES = 1  # 1 if you want to get line approximation in df


# TODO get position in ortho from image name
def get_xy_mask(name):
    s = []
    s = name.split(sep='_')
    x = int(s[-2])
    y = int(s[-1])
    return x, y


# morphological transformation
# uses dilation and erosion based on the input chosen, i.e., input other than 0 enables that particular kernel
def _morph(img, kernel_er1, kernel_di1, kernel_er2, kernel_di2):
    if str(kernel_er1) != '0':
        img = cv2.erode(img, kernel_er1, iterations=1)
    if str(kernel_di1) != '0':
        img = cv2.dilate(img, kernel_di1, iterations=1)
    if str(kernel_er2) != '0':
        img = cv2.erode(img, kernel_er2, iterations=1)
    if str(kernel_di2) != '0':
        img = cv2.dilate(img, kernel_di2, iterations=1)
    # print('morph done.')
    return img


# This function is used to calculate the centroid of the contours
def center(contour):
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


# An explanation of the working of this function would be helpful.
def fit_line(crop_bin_imgs):
    line_xy = []
    line_lengths = []
    for crop_bin_img in crop_bin_imgs:
        # find contour
        crop_bin_img, contours, hierarchy = cv2.findContours(crop_bin_img, 1, 2)
        cnt = contours[0]
        # fit line and get coordinates - NOT STABLE
        rows, cols = crop_bin_img.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        if vx > 0:
            # create black image with white line
            image_line = np.zeros((crop_bin_img.shape))
            img_line = cv2.line(image_line, (cols - 1, righty), (0, lefty), (170, 255, 0), 2)
            # overlay of line and contour
            img_sum = img_line + crop_bin_img
            img_sum[img_sum < 256] = 0  # turn all pixel not 425 (line+white contour)
            # find pixels in line
            line_x, line_y = np.where(img_sum > 256)
            line_xy.append([line_x, line_y])
            # find line length
            length = math.sqrt((line_x[-1] - line_x[0]) ** 2 + (line_y[-1] - line_y[0]) ** 2)
            line_lengths.append(length)
        else:
            print('No line found.')
            line_xy.append([[], []])
            line_lengths.append(0)
    return line_xy, line_lengths


def draw_contours(img_name, mask, mask_rgb, gps_trans, min_area, max_area):
    mask, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # create df
    features = {"name": [], "gps": [], "x_ortho": [], "y_ortho": [], "x_mask": [], "y_mask": [], "w_bb": [], "h_bb": [],
                "perimeter": [], "area": [], "centroid": []}
    i = 0
    f_contours = []
    for c in contours:
        features["name"].append(img_name + "_cnt_" + str(i))
        x, y, w, h = cv2.boundingRect(c)
        x_mask, y_mask = get_xy_mask(img_name)
        x_cnt = x + x_mask
        y_cnt = y + y_mask
        if gps_trans != 0:
            features["gps"].append(gps_trans * (x_cnt, y_cnt))
        else:
            features["gps"].append(0)
        features["x_ortho"].append(x_cnt)
        features["y_ortho"].append(y_cnt)
        features["x_mask"].append(x)
        features["y_mask"].append(y)
        features["w_bb"].append(w)
        features["h_bb"].append(h)
        features["perimeter"].append(cv2.arcLength(c, True))  # true is used when a closed contour is used as an input
        features["area"].append(cv2.contourArea(c))
        features["centroid"].append(center(c))
        i += 1
        if min_area < cv2.contourArea(c) < max_area:
            f_contours.append(c)

    df_feat = pd.DataFrame(features)
    # filter out small contours
    df_feat = df_feat[df_feat["area"] > min_area]
    df_feat = df_feat[df_feat["area"] < max_area]

    return df_feat


# what are the input parameters x_add and y_add ?
def crop_shadows(df, bin_img, rgb_img, x_add, y_add):
    bin_imgs = []
    rgb_imgs = []
    for i in df.index:
        if df['y_mask'][i] > y_add:
            y = df['y_mask'][i] - y_add
        else:
            y = 0
        if df['x_mask'][i] > x_add:
            x = df['x_mask'][i] - x_add
        else:
            x = 0
        bin_imgs.append(bin_img[y:y + df['h_bb'][i] + y_add, x:x + df['w_bb'][i] + x_add])
        rgb_imgs.append(rgb_img[y:y + df['h_bb'][i] + y_add, x:x + df['w_bb'][i] + x_add][:])
    return bin_imgs, rgb_imgs


def get_shadows(
        bin_mask,
        image,
        image_name,
        trans_matrix,
        KER_ER1=np.ones((3, 3), np.uint8),
        KER_DI1=np.ones((30, 30), np.uint8),
        KER_ER2=np.ones((7, 7), np.uint8),
        KER_DI2=0,
        X_ADD=50,
        Y_ADD=50,
        MIN_AREA=4600,  # smaller is often grass
        MAX_AREA=250000  # larger is often black area outside picture
):
    """Cut out shadows after morphological transformation

    Function to do morphological transformation on binary masks.
    Afterwards, shadow contours are cut out from the mask and the
    corresponding rgb image. Information about the cut outs is stored
    in a dataframe.

    Parameters
    ----------
    bin_mask : np.Array (width, height, 1)
        Binary mask of shadows. White pixels are 1 and black pixels are 0. 
        Size is equal to image size.
    image : np.Array (width, height, 3)
        Corresponding RGB-image for binary mask. Used to cut out shadows
        that need to be classified.
    image_name : str
        String of image name, following the convention...
    trans_matrix : 
        Transformation matrix with coordinates for...
    DEFAULT
    ???

    Returns
    -------
    bin_cuts : list of np.Array [[w, h, 1],...]
        list of the shadows that were cropped out from binary mask
        → useful for possible 3D reconstruction
    rgb_cuts : list of np.Array [[w, h, 3],...]
        list of the shadows that were cropped out from rgb image 
        → used for classification
    df_morph : pd.DataFrame
        dataframe with information on shadow contours in the same 
        order as list (area, bounding box, perimeter etc.) 
    """

    # mask_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    # ret,mask_bin = cv2.threshold(mask_gray,127,255,cv2.THRESH_BINARY)

    # morphological transformation
    mask_morph = _morph(bin_mask, KER_ER1, KER_DI1, KER_ER2, KER_DI2)
    # extract contour df
    df_cnt = draw_contours(image_name, mask_morph, image, trans_matrix, MIN_AREA, MAX_AREA)
    # cut out contours
    bin_cuts, rgb_cuts = crop_shadows(df_cnt, mask_morph, image, X_ADD, Y_ADD)
    # get lines
    # where_lines, line_lengths = fit_line(bin_cuts)
    # df_cnt["line"] = where_lines
    # df_cnt["l_length"] = line_lengths

    return bin_cuts, rgb_cuts, df_cnt

def classify_shadows(rgb_shadows, df_shadows, weights_path):
    # load trained model with weights
    os.chdir(weights_path)
    model = tf.keras.applications.DenseNet121(include_top=True, classes=3, pooling=None)
    model.load_weights(weights_path)
    # predict rgb_cuts with loaded model
    classes = []
    confidence_scores = []
    predictions = model.predict(rgb_shadows)
    score = tf.nn.softmax(predictions[0])
    # class + confidence in liste speichern
    # results als neue colums dem df hinzufuegen
    return df_shadows

if __name__ == "__main__":
    bin_mask = (cv2.imread(image_name, cv2.IMREAD_GRAYSCALE) / 255).astype(np.uint8)
    image = cv2.imread(image_name)
    # print(bin_img.shape)
    # print(np.unique(bin_img))
    # plt.imshow(bin_img)
    # plt.show()
    image_name = image_name[:-4]
    bin_cuts, rgb_cuts, df_morph = get_shadows(bin_mask, image, image_name, GPS_TRANS)
    print(df_morph)
    plt.imshow(rgb_cuts[0])
    plt.show()
