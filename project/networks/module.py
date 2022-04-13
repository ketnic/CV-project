from enum import Enum

from networks.classifier.ImageClassifier import ImageClassifier
from networks.siamese.ImageSimilarity import ImageSimilarity
from networks.siamese.config.test_config import test_config as config_s

from utils.tools.log import log
from utils.tools.convertible import list_of_tuples_to_dict
from utils.read_and_process_image import read_image


class ModuleScheme(Enum):
    onlyClassifier=1
    onlySiamese=2
    ClassifierSiamese=3
    SiameseClassifier=4


def group_images(image_paths, scheme, progress_callback=None, img_c=None, img_sim=None):
    if not img_c:
        img_c = ImageClassifier()
    if not img_sim:
        img_c = ImageSimilarity()
    
    if scheme == ModuleScheme.onlyClassifier:
        return _group_images_1(image_paths, img_c, progress_callback)
    if scheme ==  ModuleScheme.onlySiamese:
        return _group_images_2(image_paths, img_sim, progress_callback)
    if scheme == ModuleScheme.ClassifierSiamese:
        return _group_images_3(image_paths, img_c, img_sim, progress_callback)
    if scheme == ModuleScheme.SiameseClassifier:
        return _group_images_4(image_paths, img_c, img_sim, progress_callback)


def _group_images_1(image_paths, img_c, progress_callback=None):
    log('image classifying...')
    images = map(lambda image_path: read_image(img_path=image_path), image_paths)
    images = list(images)
    classes = img_c.classify_images(images=images)
    class_images = zip(classes, list(zip(image_paths, images)))
    class_images = list_of_tuples_to_dict(class_images)

    if progress_callback:
        progress_callback(0.5)

    log('image clustering...')
    count = len(class_images.keys())
    result = {}
    for i, (key, tuples) in enumerate(class_images.items()):
        image_paths = [x for x, _ in tuples]
        images      = [x for _, x in tuples]
        result[key] = { 0: image_paths }
        if progress_callback:
            progress_callback(0.5 + ( (i + 1) / count) / 2)

    return result


def _group_images_2(image_paths, img_sim, progress_callback=None):
    log('image clustering...')
    images = map(lambda image_path: read_image(img_path=image_path), image_paths)
    images = list(images)
    d_m = img_sim.compare_images(images)
    clustering = config_s.CLUSTERER.get(config_s.DECISION_THRESHOLD)
    clustering.fit(d_m)
    labels = clustering.labels_
    cluster_images = zip(labels, list(zip(image_paths, images)))
    cluster_images = list_of_tuples_to_dict(cluster_images)
    if progress_callback:
        progress_callback(0.5)

    log('image classifying...')
    count = len(cluster_images.keys())
    result = {}
    for i, (key, tuples) in enumerate(cluster_images.items()):
        image_paths = [x for x, _ in tuples]
        images      = [x for _, x in tuples]
        result[key] = { 0: image_paths }
        if progress_callback:
            progress_callback(0.5 + ( (i + 1) / count) / 2)
    return result



def _group_images_3(image_paths, img_c, img_sim, progress_callback=None):
    log('image classifying...')
    images = map(lambda image_path: read_image(img_path=image_path), image_paths)
    images = list(images)
    classes = img_c.classify_images(images=images)
    class_images = zip(classes, list(zip(image_paths, images)))
    class_images = list_of_tuples_to_dict(class_images)

    if progress_callback:
        progress_callback(0.5)

    log('image clustering...')
    count = len(class_images.keys())
    result = {}
    for i, (key, tuples) in enumerate(class_images.items()):
        image_paths = [x for x, _ in tuples]
        images      = [x for _, x in tuples]
        d_m = img_sim.compare_images(images)
        clustering = config_s.CLUSTERER.get(config_s.DECISION_THRESHOLD)
        clustering.fit(d_m)
        labels = clustering.labels_
        result[key] = list_of_tuples_to_dict(zip(labels, image_paths))
        if progress_callback:
            progress_callback(0.5 + ( (i + 1) / count) / 2)

    return result


def _group_images_4(image_paths, img_c, img_sim, progress_callback=None):
    log('image clustering...')
    images = map(lambda image_path: read_image(img_path=image_path), image_paths)
    images = list(images)
    d_m = img_sim.compare_images(images)
    clustering = config_s.CLUSTERER.get(config_s.DECISION_THRESHOLD)
    clustering.fit(d_m)
    labels = clustering.labels_
    cluster_images = zip(labels, list(zip(image_paths, images)))
    cluster_images = list_of_tuples_to_dict(cluster_images)
    if progress_callback:
        progress_callback(0.5)

    log('image classifying...')
    count = len(cluster_images.keys())
    result = {}
    for i, (key, tuples) in enumerate(cluster_images.items()):
        image_paths = [x for x, _ in tuples]
        images      = [x for _, x in tuples]
        classes = img_c.classify_images(images=images)
        result[key] = list_of_tuples_to_dict(zip(classes, image_paths))
        if progress_callback:
            progress_callback(0.5 + ( (i + 1) / count) / 2)
    return result
