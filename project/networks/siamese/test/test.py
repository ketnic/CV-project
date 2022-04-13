import os
from sklearn import metrics

from networks.siamese.config.test_config import test_config
from networks.siamese.ImageSimilarity import ImageSimilarity

from utils.tools.log import log
from utils.tools.convertible import join_to_path, list_of_tuples_to_dict, flatten
from utils.read_and_process_image import read_image


def test(config=test_config):
    flats = os.listdir(config.SIMILAR_PHOTOS)

    img_sim = ImageSimilarity(config)

    rand = []
    a_rand = []
    f_w = []
    v = []
    i = 0
    count = len(flats)
    for flat_id in flats:
        i += 1
        print()
        print(f'flats {i}/{count}')
        pred_labels, true_labels = _test(flat_id, config, img_sim.compare_images)
        rand.append(metrics.rand_score(true_labels, pred_labels))
        a_rand.append(metrics.adjusted_rand_score(true_labels, pred_labels))
        f_w.append(metrics.fowlkes_mallows_score(true_labels, pred_labels))
        v.append(metrics.v_measure_score(true_labels, pred_labels))
    print(f'rand: {sum(rand) / len(rand)}')
    print(f'a_rand: {sum(a_rand) / len(a_rand)}')
    print(f'f_w: {sum(f_w) / len(f_w)}')
    print(f'v: {sum(v) / len(v)}')


def _test(flat_id, config, compare_images):
    log('getting images with clusters...')
    flat_dir = join_to_path([config.SIMILAR_PHOTOS, flat_id])
    test_clusters = list(map(lambda x: join_to_path([flat_dir, x]), os.listdir(flat_dir)))
    test_images_in_clusters = list(map(
        lambda cluster_path: (
            os.path.basename(cluster_path),
            list(map(lambda img_name: join_to_path([cluster_path, img_name]), os.listdir(cluster_path)))
        ),
        test_clusters
    ))
    test_images_in_clusters = list_of_tuples_to_dict(test_images_in_clusters)
    test_images_in_clusters = {k: v[0] for k, v in test_images_in_clusters.items()}
    img_paths = flatten(test_images_in_clusters.values())
    img_cluster = {}
    for cluster, images in test_images_in_clusters.items():
        for image in images:
            img_cluster[image] = cluster
    images = map(lambda img_path: read_image(img_path=img_path), img_paths)

    log('comparing...')
    distance_matrix = compare_images(images=images)

    log('clustering...')
    clustering = config.CLUSTERER.get(config.DECISION_THRESHOLD)
    clustering.fit(distance_matrix)
    pred_labels = clustering.labels_
    true_labels = [img_cluster[x] for x in img_paths]

    return pred_labels, true_labels