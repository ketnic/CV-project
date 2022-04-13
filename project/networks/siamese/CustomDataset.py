import os
import cv2
import tensorflow_datasets as tfds


class CustomDataset(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }

  def __init__(self, path):
    super().__init__()

    self.path = path

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      features=tfds.features.FeaturesDict({
        'input_1_left': tfds.features.Image(shape=(224,224,3)),
        'input_2_right': tfds.features.Image(shape=(224,224,3)),
        'label': tfds.features.ClassLabel(names=['0', '1']),
      }),
      supervised_keys=('input_1_left', 'input_2_right'),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    return {
      'train': self._generate_examples(path=self.path),
    }

  def _generate_examples(self, path):
    for value in os.listdir(path):
      value_path = os.path.sep.join([path, value])
      for examples in os.listdir(value_path):
        examples_path = os.path.sep.join([value_path, examples])
        left, right = os.listdir(examples_path)
        left_path = os.path.sep.join([examples_path, left])
        right_path = os.path.sep.join([examples_path, right])
        yield value + '_' + examples, {
          'input_1_left': cv2.imread(left_path),
          'input_2_right': cv2.imread(right_path),
          'label': value,
        }
