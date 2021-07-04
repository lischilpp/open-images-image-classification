from pathlib import Path

# options

# image download

MINIMUM_CONFIDENCE_FOR_LABEL = 0.9
MAXIMUM_DOWNLOADED_IMAGE_COUNT_PER_CLASS = 100
DOWNLAD_IMAGES_HIGH_RESOLUTION = True

DIRPATH_IMAGE_DOWNLOAD = Path('/media/linus/ML/open_images')

# model training

MODEL_URL, MODEL_INPUT_SIZE = ('https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2', 380)
# MODEL_URL, MODEL_INPUT_SIZE = ("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4", 224)
# MODEL_URL, MODEL_INPUT_SIZE = ("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4", 299)

TEST_DATA_PERCENTAGE = 0.2
VALIDATION_DATA_PERCENTAGE = 0.2
DO_DATA_AUGMENTATION = True
DO_FINE_TUNING = False
DATASET_BATCH_SIZE = 32
DROPOUT_RATE = 0.2
TRAINING_EPOCHS = 32
ENABLE_CHECKPOINTS = True
ENABLE_EARLY_STOPPING = False
ENABLE_AUTOMATIC_CLASS_WEIGHTS = True

DIRPATH_DATASET = DIRPATH_IMAGE_DOWNLOAD / 'animals'

# other paths

DIRPATH_IN = Path('in')
DIRPATH_IDS = DIRPATH_IN / 'ids'
DIRPATH_LABELS = DIRPATH_IN / 'labels'
DIRPATH_BOUNDING_BOXES = DIRPATH_IN / 'boxes'
DIRPATH_CLASS_LISTS_TO_DOWNLOAD = DIRPATH_IN / 'class_lists'
FILEPATH_CLASS_NAMES = DIRPATH_IN / 'oidv6-class-descriptions.csv'

DIRPATH_PROCESSING = Path('processing')
FILEPATH_CLASS_ID_TO_IMAGE_IDS = DIRPATH_PROCESSING / 'class_id_to_image_ids.csv'

DIRPATH_OUT = Path('out')
FILEPATH_CLASS_LIST_BY_IMAGE_COUNT = DIRPATH_OUT / 'class_list_by_image_count.csv'
FILEPATH_SAVED_MODEL = DIRPATH_OUT / 'saved_model'
FILEPATH_CLASS_ACCURACIES = DIRPATH_OUT / 'accuracies.csv'
