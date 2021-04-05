from final_model import MODELS
import cv2
import rasterio as rio
import matplotlib.pyplot as plt

def main():
    # for testing purposes
    img_path = "./test_data/zcropped_2000_3005.jpg"
    weights_path = "./test_data/epochs_50_lr_0.0001_eps_1e-07_batch_8_pool_Noneweights_16_0.99.hdf5"
    filepath_knn = "./test_data/labels.csv"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_name = img_path[16:-4]
    trans_matrix = rio.open('./test_data/ortho.tif').transform

    m = MODELS(filepath_knn , weights_path)
    img_test, df_test = m.model_predict(image, image_name, trans_matrix)
    print(df_test[['class', 'confidence_score']])
    plt.imshow(img_test)
    plt.show()

if __name__ == "__main__":
    main()
