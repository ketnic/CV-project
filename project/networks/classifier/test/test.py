import os

from networks.classifier.config.test_config import test_config
from networks.classifier.ImageClassifier import ImageClassifier

from utils.tools.log import log
from utils.read_and_process_image import read_image


def test(config=test_config):
    log('loading images...')
    test_dir = r'C:\Users\NedStar\Desktop\data\test'
    images_basenames = os.listdir(test_dir)
    images_paths = list(map(lambda image_basename: os.path.sep.join([test_dir, image_basename]), images_basenames))
    images = map(lambda image_path: read_image(img_path=image_path), images_paths)
    images = list(images)
    
    log('classifying...')
    result = ImageClassifier(config).classify_images(images=images)

    print(result)
