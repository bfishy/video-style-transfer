import os
from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from six.moves import urllib
import imageio


def download(download_link, file_name):
    """
    Download a file given its URL
    """
    if os.path.exists(file_name):
        return
    print("Downloading the VGG pre-trained model. This might take a while ...")
    file_name, _ = urllib.request.urlretrieve(download_link, file_name)
    file_stat = os.stat(file_name)
    print('Successfully downloaded VGG-19 pre-trained weights.')


def get_resized_image(img_path, width, height):
    """
    Load and resize an image into the desired height and width
    """
    image = Image.open(img_path)
    # PIL is column major
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)

    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)


def generate_noise_image(content_image, width, height, noise_ratio=0.6):
    """
    Randonly generate noise in an image
    """
    noise_image = np.random.uniform(-20, 20, (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)


def save_image(path, image):
    image = image[0] # tensor to image
    image = np.clip(image, 0, 255).astype('uint8')
    imageio.imwrite(path, image)