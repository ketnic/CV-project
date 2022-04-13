import os


SOURCE_DATA = os.path.sep.join([os.path.dirname(os.getcwd()), 'data'])
SOURCE_PHOTOS = os.path.sep.join([SOURCE_DATA, 'source'])
FLAT_TYPES_CSV = os.path.sep.join([SOURCE_DATA, 'classes.csv'])
SIMILAR_PHOTOS = os.path.sep.join([SOURCE_DATA, 'similarity', 'photos'])
MERGED_DATASET_CSV = os.path.sep.join([SOURCE_DATA, 'ds.csv'])


DATA = os.path.sep.join([os.getcwd(), 'data'])
MODELS = os.path.sep.join([os.getcwd(), 'models'])
