from networks.siamese.config.test_config import test_config
from networks.siamese.ImageSimilarity import ImageSimilarity
from networks.siamese.create_datesets import create_datasets
from networks.siamese.create_auc_roc import create_auc_roc

from utils.tools.log import log
from utils.plots.auc_threshold import plot_auc_threshold

def test_auc_roc(config=test_config):
    img_sim = ImageSimilarity(config)

    log('creating datasets...')
    _, val_ds = create_datasets(
        source=config.SIMILAR_PHOTOS, 
        dist=config.DATA, 
        balanced=config.BALANCED,
        image_size=config.IMAGE_SHAPE[:-1],
        dataset_dir=config.DATASET_DIR,
        validation_split=config.VALIDATION_SPLIT,
        paired_with_itself=config.MAKE_ITSELF_PAIRS
    )

    log('auc roc...')
    thresholds = [x / 100 for x in range(1, 100)]
    auc, _, _ = create_auc_roc(img_sim.model, val_ds, decision_threshold=thresholds)

    plot_auc_threshold(thresholds, auc).show()
