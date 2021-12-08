import numpy as np
import os
from tensorflow import expand_dims
from tensorflow.keras.utils import load_img, img_to_array


def get_data():

    path = os.path.join(os.path.dirname(__file__), "images/")

    data = []

    for image in os.listdir(path):
        image_path = path + image
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = expand_dims(img_array, 0)
        data.append(img_array)

    return np.vstack(data)



if __name__ == '__main__':
    df = get_data()
