import numpy as np
from tensorflow import expand_dims
from tensorflow import image
from BloodCellAnalyzer.rbc_seg import cell_crop
from tensorflow import expand_dims


def img_seg(list_ROI):

    """Returns nd_array of shape (number of RBC on image, 128,128,1)"""

    img_list = []

    for img in list_ROI[1:]:
        img_re = image.rgb_to_grayscale(img)
        img_re = image.resize(img_re, [128,128])
        img_re = expand_dims(img_re, 0)
        img_list.append(img_re)

    img_array = np.vstack(img_list)

    return img_array

if __name__ == '__main__':
    img_array = img_seg()
