################################
# Input / output parameters
################################

IMAGES_BASE_FOLDER = './all-dogs'
LABEL_BASE_FOLDER = './Annotation'
OUTPUT_ZIP_NAME = './images.zip'


################################
# Model Hyperparameters
################################

WEIGHT_INIT_STDDEV = 5e-4
BATCH_SIZE = 128
LR_D, LR_G = 1e-3, 1e-3
EPOCHS = 2000
IMAGE_DIM = 64
NOISE_DIM = 100