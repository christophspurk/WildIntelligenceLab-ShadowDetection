
import cv2

from skimage.color import label2rgb
from skimage.measure import label

def classification_visualization(dataframe, bin_img, rgb_img):

    # label the shadows
    bin_img_gray = label(bin_img)
    image_label_overlay = label2rgb(bin_img_gray, rgb_img, bg_label = 0, kind='overlay')
    
    # TODO: CHECK end pic if its still blue
    image_label_overlay[bin_img == 0]= 0
    image_label_overlay[bin_img == 0]= rgb_img[bin_img == 0]/255
    #rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    z = dataframe

    for i, rows in z.iterrows():
        z.iloc[i]["x_mask"]
        rgb_img = cv2.rectangle(image_label_overlay, (z.iloc[i]["x_mask"], z.iloc[i]["y_mask"]), (z.iloc[i]["x_mask"]+z.iloc[i]["w_bb"], z.iloc[i]["y_mask"]+z.iloc[i]["h_bb"]), (255,0,0), 5)
        cv2.putText(image_label_overlay, str(z.iloc[i]["class"]), (z.iloc[i]["x_mask"], z.iloc[i]["y_mask"]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2)
        center_coordinates = (int(z.iloc[i]["x_mask"]+ 0.5 * z.iloc[i]["w_bb"]), int(z.iloc[i]["y_mask"]+ 0.5 * z.iloc[i]["h_bb"]))
        cv2.circle(rgb_img, center_coordinates, 10, (255, 0, 0), 10)
    
    return rgb_img