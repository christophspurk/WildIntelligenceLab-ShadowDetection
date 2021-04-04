# -*- coding: utf-8 -*-
import math
import cv2

import numpy as np
import pandas as pd



def _morph(bin_mask, kernel_er1, kernel_di1, kernel_er2, kernel_di2):
    '''morphological transformation
    uses dilation and erosion based on the input chosen,
    i.e., input other than 0 enables that particular kernel
    Parameters
    ----------
    bin_mask : np.Array (width, height, 1)
        Binary mask of shadows. White pixels are 1 and black pixels are 0.
    kernel_er1, kernel_di1, kernel_er2, kernel_di2 : np.Array
        Kernels for morph transformation. If == 0, kernel will be ignored.
        By activating the respective kernel, the order and amplitude of
        the morphological transformation can be targeted.
    Return
    ------
    bin_mask : np.Array (width, height, 1)
        Binary mask of shadows in white. Single pixels are erased by
        erosion and small contour parts are combined into a large
        contour by dilation.
    '''

    if str(kernel_er1) != '0':
        bin_mask = cv2.erode(bin_mask, kernel_er1, iterations=1)
    if str(kernel_di1) != '0':
        bin_mask = cv2.dilate(bin_mask, kernel_di1, iterations=1)
    if str(kernel_er2) != '0':
        bin_mask = cv2.erode(bin_mask, kernel_er2, iterations=1)
    if str(kernel_di2) != '0':
        bin_mask = cv2.dilate(bin_mask, kernel_di2, iterations=1)
    return bin_mask


def _get_xy_slice(image_name):
    '''
    extract position of bin_mask slice in ortho from the image name
    Parameters
    ----------
    image_name : str
        image name, following the convention: blablabla_x_y
        The blablabla part can be anything. x and y are the coordinates
        of the image or mask slice in the origin ortho.
    Returns
    -------
    x : int
        x position of the bin_mask in ortho
    y : int
        y position of the bin_mask in ortho
    '''

    s = []
    s = image_name.split(sep='_')
    x = int(s[-2])
    y = int(s[-1])
    return x, y


def _center(contour):
    '''calculate the centroid of the contours
    Parameters
    ----------
    contour : np.Array of shape (_, 1, 2)
        specifying a closed loop in a picture, here specifying the contours
        of the shadows in a binary mask

    Returns
    -------
    cx : int
        x coordinate of contour center
    cy : int
        y coordinate of contour center
    '''

    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def _analyse_contours(img_name, mask_morph, trans_matrix, min_area, max_area):
    '''Extract and analyse contours from mask
    Get the contours in the morphological transformed mask. Summarize
    information about contours in a dataframe including name, gps
    coordinates, xy-coordinates in original ortho and slice, width and
    height of contour bounding box, perimeter, area, and centroid.
    Small and large contours are filtered out as too small contours are
    usually grass and too large contours are areas outside the slice.
    Parameters
    ----------
    image_name : str
        image name, following the convention: blablabla_x_y
        The blablabla part can be anything. x and y are the coordinates
        of the image or mask slice in the origin ortho.
    mask_morph : np.Array of shape (width, height, 1)
        Binary mask of shadows in white that was already
        morphological transformed.
    trans_matrix :  Affine-matrix
        Transformation matrix with coordinates of the original ortho
        that was used to cut out mask and image slices. If it is given
        as 0, gps coordinates will only be returned as 0.
    min_area : int
        min area for contours to be relevant, smaller is often grass
    max_area : int
        max area for contours to be relevant, larger is often
        black area outside of the picture
    Returns
    -------
    df_feat : pd.DataFrame
        DataFrame of contour analysis data incl. bounding box.
    '''

    # create dict as basis for dataFrame
    features = {"name": [], "gps": [], "x_ortho": [], "y_ortho": [],
                "x_mask": [], "y_mask": [], "w_bb": [], "h_bb": [],
                "perimeter": [], "area": [], "centroid": []}
    # get contours
    contours, hierarchy = cv2.findContours(mask_morph,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # analyse contours
    i = 0
    for c in contours:
        # get coordinates
        x, y, w, h = cv2.boundingRect(c)
        x_slice, y_slice = _get_xy_slice(img_name)
        # get coordinates in ortho
        x_ortho = x + x_slice
        y_ortho = y + y_slice
        # append to lists in dict
        features["name"].append(img_name + "_cnt_" + str(i))
        if trans_matrix != 0:
            features["gps"].append(trans_matrix * (x_ortho, y_ortho))
        else:
            features["gps"].append(0)
        features["x_ortho"].append(x_ortho)
        features["y_ortho"].append(y_ortho)
        features["x_mask"].append(x)
        features["y_mask"].append(y)
        features["w_bb"].append(w)
        features["h_bb"].append(h)
        # true is used when a closed contour is used as an input
        features["perimeter"].append(cv2.arcLength(c, True))
        features["area"].append(cv2.contourArea(c))
        features["centroid"].append(_center(c))
        i += 1
    # convert dict into df
    df_feat = pd.DataFrame(features)
    # filter out to small and to large contours
    df_feat = df_feat[df_feat["area"] > min_area]
    df_feat = df_feat[df_feat["area"] < max_area]
    return df_feat


def _crop_shadows(df, bin_mask, image, x_add, y_add, x_side, y_side):
    '''Crop shadows from bin mask and image
    Crops shadows from mask and image and adds space to bounding box to
    increase recognizability for humans.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of contour analysis data incl. bounding box.
    bin_mask : np.Array (width, height, 1)
        Binary mask of shadows. White pixels are 1 and black pixels are 0.
        Size is equal to image size.
    image : np.Array (width, height, 3)
        Corresponding RGB-image for binary mask. Used to cut out shadows
        that need to be classified.

    x_add : int
        add space on the left or right of the contour bounding
        box for better recognizability of the plant in the cut out
    y_add : int
        add space on the top or buttom of the contour bounding
        box for better recognizability of the plant in the cut out
    x_side : str ('left' or 'right')
        side to which the space is added
    y_side : str ('top' or 'buttom')
        side to which the space is added

    Raises
    ------
    ValueError
        y_side must be correctly specified as 'top' or 'buttom'
    ValueError
        x_side must be correctly specified as 'left' or 'right'

    Returns
    -------
    bin_cuts : list of np.Array of shape (w, h, 1)
        list of the shadow contours that were cropped out
        from binary mask
    rgb_cuts : list of np.Array of shape (w, h, 3)
        list of the shadows that were cropped out from rgb image
    '''

    # get maximal xy-coordinates
    x_max, y_max = bin_mask.shape

    bin_cuts = []
    rgb_cuts = []
    # check for all contours in the dataFrame
    for i in df.index:
        # adjust top left coordinates of the contour bounding box
        # top or buttom
        if y_side == 'top':
            if df['y_mask'][i] > y_add:
                y = df['y_mask'][i] - y_add
            else:
                y = 0
        elif y_side == 'buttom':
            y = df['y_mask'][i]
        else:
            raise ValueError("y_side must be specified as 'top' or 'buttom'")
        # left or right
        if x_side == 'left':
            if df['x_mask'][i] > x_add:
                x = df['x_mask'][i] - x_add
            else:
                x = 0
        elif x_side == 'right':
            y = df['x_mask'][i]
        else:
            raise ValueError("x_side must be specified as 'left' or 'right'")
        # adjust bounding box heigth or width
        # heigth
        if (y + df['h_bb'][i] + y_add) < y_max:
            y2 = y + df['h_bb'][i] + y_add
        else:
            y2 = y_max
        # width
        if (x + df['w_bb'][i] + x_add) < x_max:
            x2 = x + df['w_bb'][i] + x_add
        else:
            x2 = x_max

        # cut out contours from binary masks and rgb image
        bin_cuts.append(bin_mask[y:y2, x:x2])
        rgb_cuts.append(image[y:y2, x:x2][:])
    return bin_cuts, rgb_cuts


def _fit_line(bin_cuts):
    '''line fitting

    fits line to contour in every cut out and returns
    coordinates and length
    Parameters
    ----------
    bin_cuts : list of np.Array of shape (w, h, 1)
        list of the shadow contours that were cropped out
        from binary mask
    Returns
    -------
    line_xy : list of np.Array of shape (2, _)
        list of coordinates of approximated line for contour
        in every cut out
    line_lengths: list of float
        list of length of approximated line for contour in
        every cut out
    '''

    i = 0
    line_xy = []
    line_lengths = []
    for bin_cut in bin_cuts:
        # find contour in cut out image
        contours, hierarchy = cv2.findContours(bin_cut, 1, 2)
        cnt = contours[0]
        '''fit line to contour and get coordinates
        this part is taken from the documentation. It is not fully 
        clear how it works. Other approaches did not work.'''
        rows, cols = bin_cut.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        # the code does not work for vx <= 0, it this is the case
        # no line is returned
        if vx > 0:
            # create black image with white line
            image_line = np.zeros((bin_cut.shape))
            img_line = cv2.line(image_line, (cols - 1, righty), (0, lefty),
                                (170, 255, 0), 2)
            # overlay line image and contour image
            img_sum = img_line + bin_cut
            # find pixels in line and add to list
            line_x, line_y = np.where(img_sum > 170)
            line_xy.append(np.array([line_x, line_y]))
            # find line length and add to list
            length = math.sqrt((line_x[-1] - line_x[0]) ** 2
                               + (line_y[-1] - line_y[0]) ** 2)
            line_lengths.append(length)
        else:
            print(f'No line found for contour on index {i}.')
            line_xy.append([[], []])
            line_lengths.append(0)
        i += 1
    return line_xy, line_lengths


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
        X_SIDE='left',
        Y_SIDE='top',
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
    bin_mask : np.Array of shape (width, height, 1)
        Binary mask of shadows. White pixels are 1 and black pixels are 0.
        Size is equal to image size.
    image : np.Array of shape (width, height, 3)
        Corresponding RGB-image for binary mask. Used to cut out shadows
        that need to be classified.
    image_name : str
        image name, following the convention: blablabla_x_y
        The blablabla part can be anything. x and y are the coordinates
        of the image or mask slice in the origin ortho.
    trans_matrix : Affine-matrix (default = 0)
        Transformation matrix with coordinates of the original ortho
        that was used to cut out mask and image slices. If it is given
        as 0, gps coordinates will only be returned as 0.

    KER_ER1 : np.Array (default = np.ones((3, 3), np.uint8))
    KER_DI1 : np.Array (default = np.ones((30, 30), np.uint8))
    KER_ER2 : np.Array (default = np.ones((7, 7), np.uint8))
    KER_DI2 : np.Array (default = 0)
        Kernels for morph transformation. If == 0, kernel will be ignored.
        By activating the respective kernel, the order and amplitude of
        the morphological transformation can be targeted.
    X_ADD : int (default = 0)
        add space on the left or right of the contour bounding
        box for better recognizability of the plant in the cut out
    Y_ADD : int (default = 0)
        add space on the top or buttom of the contour bounding
        box for better recognizability of the plant in the cut out
    X_SIDE : str (default = 'left')
        side - left or right - to which the space is added
    Y_SIDE : str (default = 'top')
        side - top or buttom - to which the space is added
    MIN_AREA : int (default = 4600)
        min area for contours to be relevant, smaller is often grass
    MAX_AREA : int (default = 25000)
        max area for contours to be relevant, larger is often
        black area outside of the picture
    Returns
    -------
    bin_cuts : list of np.Array of shape (w, h, 1)
        list of the shadow contours that were cropped out
        from binary mask
        → useful for possible 3D reconstruction
    rgb_cuts : list of np.Array of shape (w, h, 3)
        list of the shadows that were cropped out from rgb image
        → used for classification
    df_morph : pd.DataFrame
        dataframe with information on shadow contours in the same
        order as list (area, bounding box, perimeter etc.)
    """

    # morphological transformation
    mask_morph = _morph(bin_mask, KER_ER1, KER_DI1, KER_ER2, KER_DI2)
    # extract contour df
    df_cnt = _analyse_contours(image_name, mask_morph, trans_matrix,
                               MIN_AREA, MAX_AREA)
    # cut out contours
    bin_cuts, rgb_cuts = _crop_shadows(df_cnt, mask_morph, image,
                                       X_ADD, Y_ADD, X_SIDE, Y_SIDE)
    # add lines and line length to dataframe
    where_lines, line_lengths = _fit_line(bin_cuts)
    df_cnt["line"] = where_lines
    df_cnt["line_length"] = line_lengths
    # reset index
    df_cnt = df_cnt.reset_index(drop=True)
    return bin_cuts, rgb_cuts, df_cnt
