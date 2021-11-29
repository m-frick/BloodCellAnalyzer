from PIL import Image
import os


def resize(path, dim, output_path):
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path + item) and str(path + item).endswith(".png"):
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            f = f.strip(path)
            size = im.size
            ratio = float(dim) / max(size)
            new_image_size = tuple([int(x * ratio) for x in size])
            im = im.resize(new_image_size, Image.ANTIALIAS)
            imResize = im.resize((dim, dim), Image.ANTIALIAS)
            imResize.save(output_path + f + '_resized.jpg', 'JPEG', quality=90)
