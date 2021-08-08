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
import os
import platform
host_os = platform.system()
if host_os == 'Darwin':
    import click
else:
    pass
import shutil

import sys
import configparser
import numpy as np
import tensorflow as tf
from sources.VAE_preprocess_data import preprocess_run
from sources.VAE_get_settings import TrainandTest_settings, Train_Test_settings_show
import matplotlib.pyplot as plt
from copy import deepcopy
import xlsxwriter

prediction_image_outputs_PATH = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(prediction_image_outputs_PATH, 'libs'))

import scipy.ndimage
def zoomArray(inArray, finalShape, sameSum=False,
              zoomFunction=scipy.ndimage.zoom, **zoomKwargs):
    inArray = np.asarray(inArray, dtype=np.double)
    inShape = inArray.shape
    assert len(inShape) == len(finalShape)
    mults = []  # multipliers for the final coarsegraining
    for i in range(len(inShape)):
        if finalShape[i] < inShape[i]:
            mults.append(int(np.ceil(inShape[i] / finalShape[i])))
        else:
            mults.append(1)
    # shape to which to blow up
    tempShape = tuple([i * j for i, j in zip(finalShape, mults)])

    # stupid zoom doesn't accept the final shape. Carefully crafting the
    # multipliers to make sure that it will work.
    zoomMultipliers = np.array(tempShape) / np.array(inShape) + 0.0000001
    assert zoomMultipliers.min() >= 1

    # applying scipy.ndimage.zoom
    rescaled = zoomFunction(inArray, zoomMultipliers, **zoomKwargs)

    for ind, mult in enumerate(mults):
        if mult != 1:
            sh = list(rescaled.shape)
            assert sh[ind] % mult == 0
            newshape = sh[:ind] + [sh[ind] // mult, mult] + sh[ind + 1:]
            rescaled.shape = newshape
            rescaled = np.mean(rescaled, axis=ind + 1)
    assert rescaled.shape == finalShape

    if sameSum:
        extraSize = np.prod(finalShape) / np.prod(inShape)
        rescaled /= extraSize
    return rescaled


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

# check and remove the folder which dose not contain the necessary modalities before prepossessing step

def check_inputs(current_folder, settings, choice):


    erf =os.path.join(prediction_image_outputs_PATH, 'InputIssueReportfile.txt')
    f = open(erf, "a")
    host_os = platform.system()
    if host_os == 'Darwin':
        if os.path.isdir(os.path.join(settings['training_folder'], current_folder)):
            if len(os.listdir(os.path.join(settings['training_folder'], current_folder))) == 0:
                print(('Directory:', current_folder, 'is empty'))
                print('Warning: if the  directory is not going to be removed, the Training could be later stopped!')
                if click.confirm('The empty directory will be removed. Do you want to continue?', default=True):
                    f.write("The empty directory: %s has been removed from Training set!" % current_folder + os.linesep)
                    f.close()
                    shutil.rmtree(os.path.join(settings['training_folder'], current_folder), ignore_errors=True)
                    return
                return
        else:
            pass

    else:
        pass

        #return True


def overall_config():

    traintest_config = configparser.SafeConfigParser()
    traintest_config.read(os.path.join(prediction_image_outputs_PATH, 'config', 'configuration.cfg'))

    settings = TrainandTest_settings(traintest_config)
    settings['tmp_folder'] = prediction_image_outputs_PATH + '/tmp'
    settings['standard_lib'] = prediction_image_outputs_PATH + '/libs'
    # set paths taking into account the host OS
    host_os = platform.system()

    if settings['debug']:
        Train_Test_settings_show(settings)

    return settings

def lib_config(settings):

    device = str(settings['gpu_number'])
    print("DEBUG: ", device)
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ["CUDA_VISIBLE_DEVICES"] = device

def train_test_network_vae(settings):

    from sources.VAE_main_def import train_vae_model
    from sources.VAE_model import build_and_compile_models

    lib_config(settings)

    all_folders = os.listdir(settings['training_folder'])
    all_folders.sort()
    # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    for check in all_folders:
        check_inputs(check, settings, 'training')

    # update scan list after removing  the unnecessary folders before prepossessing step
    training_folders = os.listdir(settings['training_folder'])
    training_folders.sort()

    settings['train_test'] = 'training'
    settings['training_folder'] = os.path.normpath(settings['training_folder'])
    tmpx = settings['training_folder']
    all_foldersx = os.listdir(settings['cross_validation_folder'])
    all_foldersx.sort()
    # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    settings['training_folder'] = settings['cross_validation_folder']
    for check in all_foldersx:
        check_inputs(check, settings, 'training')

    cross_valid_folders = os.listdir(settings['cross_validation_folder'])
    cross_valid_folders.sort()

    settings['cross_validation_folder'] = os.path.normpath(settings['cross_validation_folder'])
    settings['training_folder'] = os.path.normpath(tmpx)
    latent_dim = 2048
    if settings['GWCM'] or settings['GUCM'] or settings['GRCM'] or settings['RCON']:
          BATCH_SIZE = 1
    else:
          BATCH_SIZE = 32
    img_rows = 400
    img_cols = 400
    train_generator, val_generator, test_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID = \
        preprocess_run(BATCH_SIZE, settings, img_rows, img_cols, 'train_test')
    print("> CNN: Starting training session")

    settings['model_saved_paths'] = os.path.join(prediction_image_outputs_PATH, 'models')
    settings['load_weights'] = False


    # --------------------------------------------------
    # initialize the CNN and train the classifier
    # --------------------------------------------------

    model = build_and_compile_models(settings)
    print('\x1b[6;30;44m' + 'Currently running TensorFlow version:' + '\x1b[0m', tf.__version__ )

    model = train_vae_model(model, train_generator, val_generator, test_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID,
                            settings, prediction_image_outputs_PATH)
    print('\x1b[6;30;44m' + '...........................................' + '\x1b[0m')
    print('\x1b[6;30;44m' + 'Training of network done successfully' + '\x1b[0m')
    print('\x1b[6;30;44m' + '...........................................' + '\x1b[0m')
    print('')
    settings['full_train'] = True
    settings['load_weights'] = True
    settings['model_saved_paths'] = os.path.join(prediction_image_outputs_PATH, 'models')
    settings['net_verbose'] = 0

    all_folders = os.listdir(settings['inference_folder'])
    all_folders.sort()
    # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    for check in all_folders:
        check_oututs(check, settings)
    settings['train_test'] = 'testing'
    all_folders = os.listdir(settings['inference_folder'])
    all_folders.sort()
    latent_dim = 2048
    BATCH_SIZE = 1
    img_rows = 400
    img_cols = 400
    model = build_and_compile_models(settings, img_rows, img_cols, latent_dim)
    test_generator = preprocess_run(BATCH_SIZE, settings, img_rows, img_cols, 'test')

    Image_save = prediction_image_outputs_PATH + '/prediction_image_outputs/img_{}.png'
    matrix_save = prediction_image_outputs_PATH +'/matrix_output/'

    prediction_models(model, test_generator, settings, matrix_save, Image_save)


    print('\x1b[6;30;41m' + 'Inference has been proceeded' + '\x1b[0m')


def train_network_vae(settings):


    from sources.VAE_main_def import train_vae_model
    from sources.VAE_model import build_and_compile_models

    lib_config(settings)

    all_folders = os.listdir(settings['training_folder'])
    all_folders.sort()
    # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    for check in all_folders:
        check_inputs(check, settings, 'training')

    # update scan list after removing  the unnecessary folders before prepossessing step
    training_folders = os.listdir(settings['training_folder'])
    training_folders.sort()

    settings['train_test'] = 'training'
    settings['training_folder'] = os.path.normpath(settings['training_folder'])
    tmpx = settings['training_folder']
    all_foldersx = os.listdir(settings['cross_validation_folder'])
    all_foldersx.sort()
    # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    settings['training_folder'] = settings['cross_validation_folder']
    for check in all_foldersx:
        check_inputs(check, settings, 'training')

    cross_valid_folders = os.listdir(settings['cross_validation_folder'])
    cross_valid_folders.sort()

    settings['cross_validation_folder'] = os.path.normpath(settings['cross_validation_folder'])
    settings['training_folder'] = os.path.normpath(tmpx)
    latent_dim = 2048
    if settings['GWCM'] or settings['GUCM'] or settings['GRCM'] or settings['RCON']:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = 32
    img_rows = 400
    img_cols = 400
    # train_generator, val_generator, test_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID = preprocess_run(BATCH_SIZE, settings, img_rows, img_cols)
    #
    train_generator, val_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID = \
        preprocess_run(BATCH_SIZE, settings, img_rows, img_cols, 'train')
    print("> CNN: Starting training session")

    settings['model_saved_paths'] = os.path.join(prediction_image_outputs_PATH, 'models')
    settings['load_weights'] = False


    # --------------------------------------------------
    # initialize the CNN and train the classifier
    # --------------------------------------------------
    test_generator = None
    model = build_and_compile_models(settings, img_rows, img_cols, latent_dim)
    print('\x1b[6;30;44m' + 'Currently running TensorFlow version:' + '\x1b[0m', tf.__version__ )

    model = train_vae_model(model, train_generator, val_generator, test_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID,
                            settings, prediction_image_outputs_PATH)
    print('\x1b[6;30;44m' + '...........................................' + '\x1b[0m')
    print('\x1b[6;30;44m' + 'Training of network done successfully' + '\x1b[0m')
    print('\x1b[6;30;44m' + '...........................................' + '\x1b[0m')
    print('')


def check_oututs(current_folder, settings, choice='testing'):

    erf =os.path.join(prediction_image_outputs_PATH, 'OutputIssueReportfile.txt')
    f = open(erf, "a")
    host_os = platform.system()
    if host_os == 'Darwin':
        if os.path.isdir(os.path.join(settings['inference_folder'], current_folder)):
            if len(os.listdir(os.path.join(settings['inference_folder'], current_folder))) == 0:
                print(('Directory:', current_folder, 'is empty'))
                print('Warning: if the  directory is not going to be removed, the Testing could be later stopped!')
                if click.confirm('The empty directory will be removed. Do you want to continue?', default=True):
                    f.write("The empty directory: %s has been removed from Testing set!" % current_folder + os.linesep)
                    f.close()
                    shutil.rmtree(os.path.join(settings['inference_folder'], current_folder), ignore_errors=True)
                    return
                return
        else:
            pass
    else:
        pass




def prediction_models(model, test_generator, settings, matrix_save, Image_save):

    if settings['GWCM']:

        print('\x1b[6;30;41m' + "Testing Weighted configuration model (WCM)-Gaussian" + '\x1b[0m')

        test_data = next(test_generator)
        (masked, mask), ori = test_data
        print('Number of testing images:', len(ori))
        dec1, dec2 = model['model'].predict([masked, mask])
        dec1 = tf.math.divide_no_nan(1.0, dec1)
        dec2 = tf.math.divide_no_nan(1.0, dec2)

        # dec1, dec2 = decoder(z1)
        dec1x = tf.math.divide_no_nan(1.0, dec1)
        dec2 = tf.math.divide_no_nan(1.0, dec2)
        dec1_dec2 = tf.math.multiply(dec1x, dec2)
        dec1x_dec2_dec1x = tf.math.multiply(dec1_dec2, dec1x)
        dec1x_dec2_dec1x_dec1x = tf.math.add(dec1x, dec1x_dec2_dec1x)
        dec1x_dec2_dec1x_2dec1x = tf.math.add(dec1x_dec2_dec1x_dec1x, dec1x_dec2_dec1x)
        dec2y = tf.math.divide_no_nan(dec1_dec2, dec1x_dec2_dec1x_2dec1x)
        # dec1_dec2 = tf.clip_by_value(dec1_dec2, 0.0, 0.5)

        dec1x_dec2y = tf.math.multiply(dec1x, dec2y)
        d1d2_top = tf.math.subtract(1.0, dec1x_dec2y)
        pred_img1x = tf.math.divide_no_nan(dec1x_dec2y, d1d2_top)



        # Clear current output and display test_done images
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 2, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,:])
            axes[1].imshow(pred_img1x[i,:,:,:])
            axes[0].set_title('Original Image')
            axes[1].set_title('Predicted Image')
            plt.savefig(Image_save.format(i + 1))
            plt.close()

        for i in range(len(ori)):
            f = matrix_save
            if os.path.exists(f + str(i + 1)) is False:
                print('creating new folder:', str(i + 1))
                os.mkdir(f + str(i + 1))

            else:
                pass

            nf = f + str(i + 1) + '/'
            rsizeArr1 = deepcopy(dec1x[i, :, :, 0])
            rescaled1 = zoomArray(rsizeArr1, finalShape=(40, 40))

            col = 0
            folder = nf + 'GWCM_Xi_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled1):
                worksheet.write_row(row, col, data)
            workbook.close()

            rsizeArr2 = deepcopy(dec2y[i, :, :, 0])
            rescaled2 = zoomArray(rsizeArr2, finalShape=(40, 40))

            col = 0
            folder = nf + 'GWCM_Xj_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled2):
                worksheet.write_row(row, col, data)
            workbook.close()

    if settings['GUCM']:
        print('\x1b[6;30;41m' + "Testing Undirected configuration model (UCM)-Gaussian" + '\x1b[0m')
        test_data = next(test_generator)
        (masked, mask), ori = test_data
        print('Number of testing images:', len(ori))
        pred_img1, pred_img2 = model['model'].predict([masked, mask])
        pred_img1 = tf.math.divide_no_nan(1.0, pred_img1)
        pred_img2 = tf.math.divide_no_nan(1.0, pred_img2)
        d1d2 = tf.math.multiply(pred_img1, pred_img2)
        d1d2n = tf.math.add(1.0, d1d2)
        pred_img2x = tf.math.divide_no_nan(d1d2, d1d2n)



        # Clear current output and display test_done images
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 2, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,:])
            axes[1].imshow(pred_img2x[i,:,:,:])
            axes[0].set_title('Original Image')
            axes[1].set_title('Predicted Image')
            plt.savefig(Image_save.format(i + 1))
            plt.close()

        for i in range(len(ori)):
            f = matrix_save
            if os.path.exists(f + str(i + 1)) is False:
                print('creating new folder:', str(i + 1))
                os.mkdir(f + str(i + 1))

            else:
                pass

            nf = f + str(i + 1) + '/'
            rsizeArr1 = deepcopy(pred_img1[i, :, :, 0])
            rescaled1 = zoomArray(rsizeArr1, finalShape=(40, 40))

            col = 0
            folder = nf + 'GUCM_Xi_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled1):
                worksheet.write_row(row, col, data)
            workbook.close()

            rsizeArr2 = deepcopy(pred_img2[i, :, :, 0])
            rescaled2 = zoomArray(rsizeArr2, finalShape=(40, 40))

            col = 0
            folder = nf + 'GUCM_Xj_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled2):
                worksheet.write_row(row, col, data)
            workbook.close()

    if settings['GRCM']:
        print('\x1b[6;30;41m' + "Testing Reciprocal configuration model (RCM)-Gaussian" + '\x1b[0m')
        test_data = next(test_generator)
        (masked, mask), ori = test_data
        print('Number of testing images:', len(ori))
        dec1x, dec2x, dec1y, dec2y, dec1z, dec2z = model['model'].predict([masked, mask])

        dec1x = tf.math.divide_no_nan(1.0, dec1x)
        dec2x = tf.math.divide_no_nan(1.0, dec2x)
        dec1y = tf.math.divide_no_nan(1.0, dec1y)
        dec2y = tf.math.divide_no_nan(1.0, dec2y)
        dec1z = tf.math.divide_no_nan(1.0, dec1z)
        dec2z = tf.math.divide_no_nan(1.0, dec2z)
        xiyj = tf.math.multiply(dec1x, dec2y)
        xjyi = tf.math.multiply(dec2x, dec1y)
        zjzi = tf.math.multiply(dec1z, dec2z)

        d1d2d3n = tf.math.add(1.0, tf.math.add(tf.math.add(xiyj, xjyi), zjzi))
        y_pred_right = tf.math.divide_no_nan(xiyj, d1d2d3n)
        y_pred_left = tf.math.divide_no_nan(xjyi, d1d2d3n)
        z_pred_left_right = tf.math.divide_no_nan(zjzi, d1d2d3n)



        # Clear current output and display test_done images
        for i in range(len(ori)):
            pred_img3 = np.zeros((400, 400, 3), dtype=np.float32)
            _, axes = plt.subplots(1, 2, figsize=(20, 5))
            pred_img3[:, :, 0] = y_pred_right[i, :, :, 0]
            pred_img3[:, :, 1] = y_pred_left[i, :, :, 1]
            pred_img3[:, :, 2] = z_pred_left_right[i, :, :, 2]
            axes[0].imshow(masked[i,:,:,:])
            axes[1].imshow(pred_img3[i,:,:,:])
            axes[0].set_title('Original Image')
            axes[1].set_title('Predicted Image')
            plt.savefig(Image_save.format(i + 1))
            plt.close()

        for i in range(len(ori)):
            f = matrix_save
            if os.path.exists(f + str(i + 1)) is False:
                print('creating new folder:', str(i + 1))
                os.mkdir(f + str(i + 1))

            else:
                pass

            nf = f + str(i + 1) + '/'
            rsizeArr1 = deepcopy(dec1x[i, :, :, 0])
            rescaled1 = zoomArray(rsizeArr1, finalShape=(40, 40))

            col = 0
            folder = nf + 'GRCM_Xi_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled1):
                worksheet.write_row(row, col, data)
            workbook.close()

            rsizeArr2 = deepcopy(dec2y[i, :, :, 0])
            rescaled2 = zoomArray(rsizeArr2, finalShape=(40, 40))

            col = 0
            folder = nf + 'GRCM_Yj_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled2):
                worksheet.write_row(row, col, data)
            workbook.close()


            rsizeArr1 = deepcopy(dec2x[i, :, :, 0])
            rescaled1 = zoomArray(rsizeArr1, finalShape=(40, 40))

            col = 0
            folder = nf + 'GRCM_Xj_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled1):
                worksheet.write_row(row, col, data)
            workbook.close()

            rsizeArr2 = deepcopy(dec1y[i, :, :, 0])
            rescaled2 = zoomArray(rsizeArr2, finalShape=(40, 40))

            col = 0
            folder = nf + 'GRCM_Yi_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled2):
                worksheet.write_row(row, col, data)
            workbook.close()


            rsizeArr1 = deepcopy(dec1z[i, :, :, 0])
            rescaled1 = zoomArray(rsizeArr1, finalShape=(40, 40))

            col = 0
            folder = nf + 'GRCM_Zi_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled1):
                worksheet.write_row(row, col, data)
            workbook.close()

            rsizeArr2 = deepcopy(dec2z[i, :, :, 0])
            rescaled2 = zoomArray(rsizeArr2, finalShape=(40, 40))

            col = 0
            folder = nf + 'GRCM_Zj_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled2):
                worksheet.write_row(row, col, data)
            workbook.close()


    if settings['UWCM']:
        print('\x1b[6;30;41m' + "Testing Weighted configuration model (WCM)-Uniform" + '\x1b[0m')
        test_data = next(test_generator)
        (ori, mask) = test_data
        print('Number of testing images:', len(ori))
        dec1, dec2 = model['model'].predict(ori)

        dec1x = tf.math.divide_no_nan(1.0, dec1)
        dec2 = tf.math.divide_no_nan(1.0, dec2)
        dec1_dec2 = tf.math.multiply(dec1x, dec2)
        dec1x_dec2_dec1x = tf.math.multiply(dec1_dec2, dec1x)
        dec1x_dec2_dec1x_dec1x = tf.math.add(dec1x, dec1x_dec2_dec1x)
        dec1x_dec2_dec1x_2dec1x = tf.math.add(dec1x_dec2_dec1x_dec1x, dec1x_dec2_dec1x)
        dec2y = tf.math.divide_no_nan(dec1_dec2, dec1x_dec2_dec1x_2dec1x)
        # dec1_dec2 = tf.clip_by_value(dec1_dec2, 0.0, 0.5)

        dec1x_dec2y = tf.math.multiply(dec1x, dec2y)
        d1d2_top = tf.math.subtract(1.0, dec1x_dec2y)
        pred_img4 = tf.math.divide_no_nan(dec1x_dec2y, d1d2_top)



        # Clear current output and display test_done images
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 2, figsize=(20, 5))
            axes[0].imshow(ori[i,:,:,:])
            axes[1].imshow(pred_img4[i,:,:,:])
            axes[0].set_title('Original Image')
            axes[1].set_title('Predicted Image')
            plt.savefig(Image_save.format(i + 1))
            plt.close()
        for i in range(len(ori)):
            f = matrix_save
            if os.path.exists(f + str(i + 1)) is False:
                print('creating new folder:', str(i + 1))
                os.mkdir(f + str(i + 1))

            else:
                pass

            nf = f + str(i + 1) + '/'
            rsizeArr1 = deepcopy(dec1x[i, :, :, 0])
            rescaled1 = zoomArray(rsizeArr1, finalShape=(40, 40))

            col = 0
            folder = nf + 'UWCM_Xi_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled1):
                worksheet.write_row(row, col, data)
            workbook.close()

            rsizeArr2 = deepcopy(dec2y[i, :, :, 0])
            rescaled2 = zoomArray(rsizeArr2, finalShape=(40, 40))

            col = 0
            folder = nf + 'UWCM_Xj_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled2):
                worksheet.write_row(row, col, data)
            workbook.close()

    if settings['UUCM']:
        print('\x1b[6;30;41m' + "Testing Undirected configuration model (UCM)-Uniform" + '\x1b[0m')
        test_data = next(test_generator)
        (ori, mask) = test_data
        print('Number of testing images:', len(ori))
        pred_img1, pred_img2 = model['model'].predict(ori)
        pred_img1 = tf.math.divide_no_nan(1.0, pred_img1)
        pred_img2 = tf.math.divide_no_nan(1.0, pred_img2)
        d1d2 = tf.math.multiply(pred_img1, pred_img2)
        d1d2n = tf.math.add(1.0, d1d2)
        pred_img5 = tf.math.divide_no_nan(d1d2, d1d2n)



        # Clear current output and display test_done images
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 2, figsize=(20, 5))
            axes[0].imshow(ori[i,:,:,:])
            axes[1].imshow(pred_img5[i,:,:,:])
            axes[0].set_title('Original Image')
            axes[1].set_title('Predicted Image')
            plt.savefig(Image_save.format(i + 1))
            plt.close()

        for i in range(len(ori)):
            f = matrix_save
            if os.path.exists(f + str(i + 1)) is False:
                print('creating new folder:', str(i + 1))
                os.mkdir(f + str(i + 1))

            else:
                 pass

            nf = f + str(i + 1) + '/'
            rsizeArr1 = deepcopy(pred_img1[i, :, :, 0])
            rescaled1 = zoomArray(rsizeArr1, finalShape=(40, 40))

            col = 0
            folder = nf + 'UUCM_Xi_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled1):
                worksheet.write_row(row, col, data)
            workbook.close()

            rsizeArr2 = deepcopy(pred_img2[i, :, :, 0])
            rescaled2 = zoomArray(rsizeArr2, finalShape=(40, 40))

            col = 0
            folder = nf + 'UUCM_Xj_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled2):
                worksheet.write_row(row, col, data)
            workbook.close()

    if settings['URCM']:
        print('\x1b[6;30;41m' + "Testing Reciprocal configuration model (RCM)-Uniform" + '\x1b[0m')
        test_data = next(test_generator)
        (ori, mask) = test_data
        print('Number of testing images:', len(ori))
        dec1x, dec2x, dec1y, dec2y, dec1z, dec2z = model['model'].predict(ori)

        dec1x = tf.math.divide_no_nan(1.0, dec1x)
        dec2x = tf.math.divide_no_nan(1.0, dec2x)
        dec1y = tf.math.divide_no_nan(1.0, dec1y)
        dec2y = tf.math.divide_no_nan(1.0, dec2y)
        dec1z = tf.math.divide_no_nan(1.0, dec1z)
        dec2z = tf.math.divide_no_nan(1.0, dec2z)
        xiyj = tf.math.multiply(dec1x, dec2y)
        xjyi = tf.math.multiply(dec2x, dec1y)
        zjzi = tf.math.multiply(dec1z, dec2z)

        d1d2d3n = tf.math.add(1.0, tf.math.add(tf.math.add(xiyj, xjyi), zjzi))
        y_pred_right = tf.math.divide_no_nan(xiyj, d1d2d3n)
        y_pred_left = tf.math.divide_no_nan(xjyi, d1d2d3n)
        z_pred_left_right = tf.math.divide_no_nan(zjzi, d1d2d3n)



        # Clear current output and display test_done images
        for i in range(len(ori)):
            pred_img6 = np.zeros((400, 400, 3), dtype=np.float32)
            _, axes = plt.subplots(1, 2, figsize=(20, 5))
            pred_img6[:, :, 0] = y_pred_right[i, :, :, 0]
            pred_img6[:, :, 1] = y_pred_left[i, :, :, 1]
            pred_img6[:, :, 2] = z_pred_left_right[i, :, :, 2]
            axes[0].imshow(ori[i,:,:,:])
            axes[1].imshow(pred_img6[i,:,:,:])
            axes[0].set_title('Original Image')
            axes[1].set_title('Predicted Image')
            plt.savefig(Image_save.format(i + 1))
            plt.close()


        for i in range(len(ori)):
            f = matrix_save
            if os.path.exists(f + str(i + 1)) is False:
                print('creating new folder:', str(i + 1))
                os.mkdir(f + str(i + 1))

            else:
                pass
            nf = f + str(i + 1) + '/'
            rsizeArr1 = deepcopy(dec1x[i, :, :, 0])
            rescaled1 = zoomArray(rsizeArr1, finalShape=(40, 40))

            col = 0
            folder = nf + 'URCM_Xi_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled1):
                worksheet.write_row(row, col, data)
            workbook.close()

            rsizeArr2 = deepcopy(dec2y[i, :, :, 0])
            rescaled2 = zoomArray(rsizeArr2, finalShape=(40, 40))

            col = 0
            folder = nf + 'URCM_Yj_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled2):
                worksheet.write_row(row, col, data)
            workbook.close()


            rsizeArr1 = deepcopy(dec2x[i, :, :, 0])
            rescaled1 = zoomArray(rsizeArr1, finalShape=(40, 40))

            col = 0
            folder = nf + 'URCM_Xj_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled1):
                worksheet.write_row(row, col, data)
            workbook.close()

            rsizeArr2 = deepcopy(dec1y[i, :, :, 0])
            rescaled2 = zoomArray(rsizeArr2, finalShape=(40, 40))

            col = 0
            folder = nf + 'URCM_Yi_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled2):
                worksheet.write_row(row, col, data)
            workbook.close()


            rsizeArr1 = deepcopy(dec1z[i, :, :, 0])
            rescaled1 = zoomArray(rsizeArr1, finalShape=(40, 40))

            col = 0
            folder = nf + 'URCM_Zi_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled1):
                worksheet.write_row(row, col, data)
            workbook.close()

            rsizeArr2 = deepcopy(dec2z[i, :, :, 0])
            rescaled2 = zoomArray(rsizeArr2, finalShape=(40, 40))

            col = 0
            folder = nf + 'URCM_Zj_' + str(i + 1) + '.xlsx'
            workbook = xlsxwriter.Workbook(folder)
            worksheet = workbook.add_worksheet()
            for row, data in enumerate(rescaled2):
                worksheet.write_row(row, col, data)
            workbook.close()


    if settings['RCON']:
        print('\x1b[6;30;41m' + "Testing reconstruction model (RCON)-Gaussian" + '\x1b[0m')
        test_data = next(test_generator)
        (masked, mask), ori = test_data
        print('Number of testing images:', len(ori))
        pred_img7 = model['model'].predict([masked, mask])

        # Clear current output and display test_done images
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 2, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,:])
            axes[1].imshow(pred_img7[i,:,:,:])
            axes[0].set_title('Original Image')
            axes[1].set_title('Predicted Image')
            plt.savefig(Image_save.format(i + 1))
            plt.close()


def infer_vae(settings):

    lib_config(settings)


    from sources.VAE_model import build_and_compile_models
    prediction_image_outputs_PATH = os.path.split(os.path.realpath(__file__))[0]

    settings['full_train'] = True
    settings['load_weights'] = True
    settings['model_saved_paths'] = os.path.join(prediction_image_outputs_PATH, 'models')
    settings['net_verbose'] = 0
    all_folders = os.listdir(settings['inference_folder'])
    all_folders.sort()
    # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    for check in all_folders:
       check_oututs(check, settings)
    settings['train_test'] = 'testing'
    all_folders = os.listdir(settings['inference_folder'])
    all_folders.sort()
    latent_dim = 2048
    if settings['GWCM'] or settings['GUCM'] or settings['GRCM'] or settings['RCON']:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = 32
    img_rows = 400
    img_cols = 400
    model = build_and_compile_models(settings, img_rows, img_cols, latent_dim)
    test_generator = preprocess_run(BATCH_SIZE, settings, img_rows, img_cols, 'test')

    Image_save = prediction_image_outputs_PATH + '/prediction_image_outputs/img_{}.png'
    matrix_save = prediction_image_outputs_PATH +'/matrix_output/'

    prediction_models(model, test_generator, settings, matrix_save, Image_save)

    print('\x1b[6;30;41m' + 'Inference has been proceeded' + '\x1b[0m')
