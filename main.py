from morph import get_shadows
from classify_shadows import classify_shadows
import cv2
import rasterio as rio
import numpy as np

# for testing purposes
def main():
    path = "./test_data/bin_transparent_mosaic_45056_6144.png"
    img_path = "./test_data/rgb_transparent_mosaic_45056_6144.png"

    # bin_mask = (cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)
    bin_mask = (cv2.imread(path) / 255)[:, :, 1].astype(np.uint8)
    image = cv2.imread(img_path)
    image_name = img_path[16:-4]
    trans_matrix = rio.open('./test_data/ortho.tif').transform

    bin_cuts, rgb_cuts, df_morph = get_shadows(bin_mask, image, image_name,
                                               trans_matrix)
    print(df_morph)


if __name__ == "__main__":
    main()