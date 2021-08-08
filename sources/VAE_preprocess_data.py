# --------------------------------------------------
#
#     Copyright (C) {2020-2021} Kevin Bronik
#     UCL Computer Science
#     https://www.ucl.ac.uk/computer-science/


#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#
#     {Variational Auto Encoder (VAE) Gaussian Uniform Boltzmann}
#     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.


import nibabel as nib


import os
import gc

from copy import deepcopy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from random import randint, seed
import numpy as np



CEND      = '\33[0m'
CBOLD     = '\33[1m'
CITALIC   = '\33[3m'
CURL      = '\33[4m'
CBLINK    = '\33[5m'
CBLINK2   = '\33[6m'
CSELECTED = '\33[7m'

CBLACK  = '\33[30m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE  = '\33[36m'
CWHITE  = '\33[37m'

CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

CGREY    = '\33[90m'
CRED2    = '\33[91m'
CGREEN2  = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2   = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2  = '\33[96m'
CWHITE2  = '\33[97m'

CGREYBG    = '\33[100m'
CREDBG2    = '\33[101m'
CGREENBG2  = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2   = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2  = '\33[106m'
CWHITEBG2  = '\33[107m'


THIS_PATH = os.path.split(os.path.realpath(__file__))[0]
Image_p = THIS_PATH + '/nozero.png'

def M_normalize(img, mask=None):
    img_data = img.get_data()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_data()
    elif mask == 'nomask':
        mask_data = img_data == img_data
    else:
        mask_data = img_data > img_data.mean()
    logical_mask = mask_data == 1  # force the mask to be logical type
    mean = img_data[logical_mask].mean()
    std = img_data[logical_mask].std()
    normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
    return normalized

def flatten_this(l):

    return flatten_this(l[0]) + (flatten_this(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

def get_mask(random_seed=None, Subject=None):
    if random_seed:
        seed(random_seed)
    if Subject == 'train':

        return mpimg.imread(Image_p)[:,:,:3]

    if Subject == 'valid':

        return mpimg.imread(Image_p)[:,:,:3]

    if Subject == 'test':

        return mpimg.imread(Image_p)[:,:,:3]

class AugmentingDataGenerator_gaussian(ImageDataGenerator):
    def flow_from_directory(self, directory, subject=None, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:

            # Get augmentend image samples
            ori = next(generator)
            # this = ori[0][:,:,0]
            # Get masks for each image sample
            mask = np.stack([
                get_mask(seed, Subject=subject)
                for _ in range(ori.shape[0])], axis=0
            )

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1
            # masked[mask==0] = np.average(this)
            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori

class AugmentingDataGenerator_uniform(ImageDataGenerator):
    def flow_from_directory(self, directory, subject=None, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:

            # Get augmentend image samples
            ori = next(generator)

            # # Get masks for each image sample
            # mask = np.stack([
            #     get_mask(seed, Subject=subject)
            #     for _ in range(ori.shape[0])], axis=0
            # )
            #
            # # Apply masks to all image sample
            # masked = deepcopy(ori)
            # masked[mask==0] = 1
            #
            # # Yield ([ori, masl],  ori) training batches
            # # print(masked.shape, ori.shape)
            gc.collect()
            yield (ori, ori)

def get_set_input_images_uniform_train_test(BATCH_SIZE, settings, img_rows, img_cols):


    TRAIN_DIR = settings['training_folder']
    VAL_DIR = settings['cross_validation_folder']
    TEST_DIR = settings['inference_folder']
    def print_info(filepath, data):
        filenames = [f for f in os.listdir(filepath + '/data')]
        input_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
        print(">> Found {} {} in {}".format(len(input_files), data, filepath))
        return len(input_files)

    trd = print_info(TRAIN_DIR, 'training data')
    vd = print_info(VAL_DIR, 'validation data')
    td = print_info(TEST_DIR, 'testing data')
    STEP_SIZE_TRAIN = int(trd//BATCH_SIZE)
    STEP_SIZE_VALID = int(vd//BATCH_SIZE)
    train_datagen = AugmentingDataGenerator_uniform(
        # rotation_range=10,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True, vertical_flip=True
        # horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        subject='train',
        target_size=(img_rows, img_cols),
        batch_size=BATCH_SIZE

    )

    # Create validation generator
    val_datagen = AugmentingDataGenerator_uniform(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        subject='valid',
        target_size=(img_rows, img_cols),
        batch_size=BATCH_SIZE

        #classes=['val'],
    )

    # Create testing generator
    test_datagen = AugmentingDataGenerator_uniform(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        subject='test',
        target_size=(img_rows, img_cols),
        batch_size=BATCH_SIZE

    )

    return train_generator, val_generator, test_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID


def get_set_input_images_uniform_train(BATCH_SIZE, settings, img_rows, img_cols):


    TRAIN_DIR = settings['training_folder']
    VAL_DIR = settings['cross_validation_folder']
    # TEST_DIR = settings['inference_folder']
    def print_info(filepath, data):
        filenames = [f for f in os.listdir(filepath + '/data')]
        input_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
        print(">> Found {} {} in {}".format(len(input_files), data, filepath))
        return len(input_files)

    trd = print_info(TRAIN_DIR, 'training data')
    vd = print_info(VAL_DIR, 'validation data')
    # td = print_info(TEST_DIR, 'testing data')
    STEP_SIZE_TRAIN = int(trd//BATCH_SIZE)
    STEP_SIZE_VALID = int(vd//BATCH_SIZE)
    train_datagen = AugmentingDataGenerator_uniform(
        # rotation_range=10,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True, vertical_flip=True
        # horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        subject='train',
        target_size=(img_rows, img_cols),
        batch_size=BATCH_SIZE

    )

    # Create validation generator
    val_datagen = AugmentingDataGenerator_uniform(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        subject='valid',
        target_size=(img_rows, img_cols),
        batch_size=BATCH_SIZE

        #classes=['val'],
    )

    # # Create testing generator
    # test_datagen = AugmentingDataGenerator_uniform(rescale=1./255)
    # test_generator = test_datagen.flow_from_directory(
    #     TEST_DIR,
    #     subject='test',
    #     target_size=(img_rows, img_cols),
    #     batch_size=BATCH_SIZE
    #
    # )

    return train_generator, val_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID




def get_set_input_images_gaussian_train_test(BATCH_SIZE, settings, img_rows, img_cols):


    TRAIN_DIR = settings['training_folder']
    VAL_DIR = settings['cross_validation_folder']
    TEST_DIR = settings['inference_folder']
    def print_info(filepath, data):
        filenames = [f for f in os.listdir(filepath + '/data')]
        input_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
        print(">> Found {} {} in {}".format(len(input_files), data, filepath))
        return len(input_files)

    trd = print_info(TRAIN_DIR, 'training data')
    vd = print_info(VAL_DIR, 'validation data')
    td = print_info(TEST_DIR, 'testing data')
    STEP_SIZE_TRAIN = int(trd//BATCH_SIZE)
    STEP_SIZE_VALID = int(vd//BATCH_SIZE)
    train_datagen = AugmentingDataGenerator_gaussian(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    subject='train',
    target_size=(img_rows, img_cols),
    batch_size=BATCH_SIZE)


# Create validation generator
    val_datagen = AugmentingDataGenerator_gaussian(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    subject='valid',
    target_size=(img_rows, img_cols),
    batch_size=BATCH_SIZE)

# Create testing generator
    test_datagen = AugmentingDataGenerator_gaussian(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    subject='test',
    target_size=(img_rows, img_cols),
    batch_size=BATCH_SIZE)

    return train_generator, val_generator, test_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID



def get_set_input_images_gaussian_train(BATCH_SIZE, settings, img_rows, img_cols):


    TRAIN_DIR = settings['training_folder']
    VAL_DIR = settings['cross_validation_folder']
    # TEST_DIR = settings['inference_folder']
    def print_info(filepath, data):
        filenames = [f for f in os.listdir(filepath + '/data')]
        input_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
        print(">> Found {} {} in {}".format(len(input_files), data, filepath))
        return len(input_files)

    trd = print_info(TRAIN_DIR, 'training data')
    vd = print_info(VAL_DIR, 'validation data')
    # td = print_info(TEST_DIR, 'testing data')
    STEP_SIZE_TRAIN = int(trd//BATCH_SIZE)
    STEP_SIZE_VALID = int(vd//BATCH_SIZE)
    train_datagen = AugmentingDataGenerator_gaussian(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        subject='train',
        target_size=(img_rows, img_cols),
        batch_size=BATCH_SIZE)


    # Create validation generator
    val_datagen = AugmentingDataGenerator_gaussian(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        subject='valid',
        target_size=(img_rows, img_cols),
        batch_size=BATCH_SIZE)

    # # Create testing generator
    # test_datagen = AugmentingDataGenerator_gaussian(rescale=1./255)
    # test_generator = test_datagen.flow_from_directory(
    #     TEST_DIR,
    #     subject='test',
    #     target_size=(img_rows, img_cols),
    #     batch_size=BATCH_SIZE)

    return train_generator, val_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID



def get_set_input_images_gaussian_test(BATCH_SIZE, settings, img_rows, img_cols):


    # TRAIN_DIR = settings['training_folder']
    # VAL_DIR = settings['cross_validation_folder']
    TEST_DIR = settings['inference_folder']
    def print_info(filepath, data):
        filenames = [f for f in os.listdir(filepath + '/data')]
        input_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
        print(">> Found {} {} in {}".format(len(input_files), data, filepath))
        return len(input_files)

    # trd = print_info(TRAIN_DIR, 'training data')
    # vd = print_info(VAL_DIR, 'validation data')
    td = print_info(TEST_DIR, 'testing data')

    # BATCH_SIZE = int(td)
    # STEP_SIZE_TRAIN = int(trd//BATCH_SIZE)
    # STEP_SIZE_VALID = int(vd//BATCH_SIZE)
    # train_datagen = AugmentingDataGenerator_gaussian(
    #     rotation_range=10,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     rescale=1./255,
    #     horizontal_flip=True
    # )
    # train_generator = train_datagen.flow_from_directory(
    #     TRAIN_DIR,
    #     subject='train',
    #     target_size=(img_rows, img_cols),
    #     batch_size=BATCH_SIZE)
    #
    #
    # # Create validation generator
    # val_datagen = AugmentingDataGenerator_gaussian(rescale=1./255)
    # val_generator = val_datagen.flow_from_directory(
    #     VAL_DIR,
    #     subject='valid',
    #     target_size=(img_rows, img_cols),
    #     batch_size=BATCH_SIZE)

    # Create testing generator
    test_datagen = AugmentingDataGenerator_gaussian(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        subject='test',
        target_size=(img_rows, img_cols),
        batch_size=int(td))

    return test_generator

def get_set_input_images_uniform_test(BATCH_SIZE, settings, img_rows, img_cols):


    # TRAIN_DIR = settings['training_folder']
    # VAL_DIR = settings['cross_validation_folder']
    TEST_DIR = settings['inference_folder']
    def print_info(filepath, data):
        filenames = [f for f in os.listdir(filepath + '/data')]
        input_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
        print(">> Found {} {} in {}".format(len(input_files), data, filepath))
        return len(input_files)

    # trd = print_info(TRAIN_DIR, 'training data')
    # vd = print_info(VAL_DIR, 'validation data')
    td = print_info(TEST_DIR, 'testing data')
    # BATCH_SIZE = int(td)
    # STEP_SIZE_TRAIN = int(trd//BATCH_SIZE)
    # STEP_SIZE_VALID = int(vd//BATCH_SIZE)
    # train_datagen = AugmentingDataGenerator_uniform(
    #     # rotation_range=10,
    #     # width_shift_range=0.1,
    #     # height_shift_range=0.1,
    #     rescale=1./255,
    #     # shear_range=0.2,
    #     # zoom_range=0.2,
    #     horizontal_flip=True, vertical_flip=True
    #     # horizontal_flip=True
    # )
    # train_generator = train_datagen.flow_from_directory(
    #     TRAIN_DIR,
    #     subject='train',
    #     target_size=(img_rows, img_cols),
    #     batch_size=BATCH_SIZE
    #
    # )
    #
    # # Create validation generator
    # val_datagen = AugmentingDataGenerator_uniform(rescale=1./255)
    # val_generator = val_datagen.flow_from_directory(
    #     VAL_DIR,
    #     subject='valid',
    #     target_size=(img_rows, img_cols),
    #     batch_size=BATCH_SIZE
    #
    #     #classes=['val'],
    # )

    # Create testing generator
    test_datagen = AugmentingDataGenerator_uniform(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        subject='test',
        target_size=(img_rows, img_cols),
        batch_size=int(td)

    )

    return test_generator





def preprocess_run(BATCH_SIZE, settings, img_rows, img_cols, sub):
    if sub == 'train_test':
        if settings['GWCM'] or settings['GUCM'] or settings['GRCM'] or settings['RCON']:
            return get_set_input_images_gaussian_train_test(BATCH_SIZE, settings, img_rows, img_cols)
        else:
            return get_set_input_images_uniform_train_test(BATCH_SIZE, settings, img_rows, img_cols)
    elif sub == 'train':
        if settings['GWCM'] or settings['GUCM'] or settings['GRCM'] or settings['RCON']:
            return get_set_input_images_gaussian_train(BATCH_SIZE, settings, img_rows, img_cols)
        else:
            return get_set_input_images_uniform_train(BATCH_SIZE, settings, img_rows, img_cols)
    elif sub == 'test':
        if settings['GWCM'] or settings['GUCM'] or settings['GRCM'] or settings['RCON']:
            return get_set_input_images_gaussian_test(BATCH_SIZE, settings, img_rows, img_cols)
        else:
            return get_set_input_images_uniform_test(BATCH_SIZE, settings, img_rows, img_cols)

    else:
        print('not supported!')
        return





