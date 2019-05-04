import torch
import PIL.Image

def load_image(file_name, size=None):
    image = PIL.Image.open(file_name)
    if size is not None:
        Image = image.resize((size, size), image.ANTIALIAS)
    return image

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram