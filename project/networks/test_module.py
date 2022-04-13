import os
import config
import pandas as pd
from sklearn import metrics

from networks.classifier.ImageClassifier import ImageClassifier
from networks.siamese.ImageSimilarity import ImageSimilarity
from networks.module import group_images, ModuleScheme

from utils.tools.log import log


def test_module():
    log('loading dataset...')
    df = pd.read_csv(config.MERGED_DATASET_CSV)
    df['photo_path'] = df.apply(lambda row: '{}\\{}\\{}.jpeg'.format(config.SOURCE_PHOTOS, row['flat_id'], row['photo_num']), axis=1)
    flats = df['flat_id'].unique()

    log('loading models...')
    img_c = ImageClassifier()
    img_sim = ImageSimilarity()

    log('testing...')
    rand = []
    a_rand = []
    f_w = []
    v = []
    i = 0
    for flat_id in flats:
        i += 1
        print(f'flats {i}/{len(flats)}')
        sub_df = df[df['flat_id'] == flat_id]
        img_paths = sub_df['photo_path'].to_numpy()
        result = group_images(img_paths, ModuleScheme.ClassifierSiamese, img_c=img_c, img_sim=img_sim)
        clusters = []
        for _, rooms in result.items():
            for _, images in rooms.items():
                clusters.append(images)
        clusters = list(zip(range(len(clusters)), clusters))
        img_cluster = {}
        for k, img_paths in clusters:
            for image in img_paths:
                img_cluster[image] = k
        true_labels = []
        pred_labels = []
        for cluster, images in clusters:
            for image in images:
                pred_labels.append(cluster)
                true_labels.append(sub_df[sub_df['photo_num'] == int(os.path.basename(image).replace('.jpeg', ''))].iloc[0, 3])
        rand.append(metrics.rand_score(true_labels, pred_labels))
        a_rand.append(metrics.adjusted_rand_score(true_labels, pred_labels))
        f_w.append(metrics.fowlkes_mallows_score(true_labels, pred_labels))
        v.append(metrics.v_measure_score(true_labels, pred_labels))
    log(f'rand: {sum(rand) / len(rand)}')
    log(f'a_rand: {sum(a_rand) / len(a_rand)}')
    log(f'f_w: {sum(f_w) / len(f_w)}')
    log(f'v: {sum(v) / len(v)}')
