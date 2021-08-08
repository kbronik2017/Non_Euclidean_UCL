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
import signal
import threading
import time
import shutil
import numpy as np
import tensorflow as tf


from .VAE_net_Gaussian_reconstruction import get_network_Gaussian_reconstruction
from .VAE_net_Gaussian_Boltzmann_WCM import get_network_Gaussian_Boltzmann_WCM
from .VAE_net_Gaussian_Boltzmann_RCM import get_network_Gaussian_Boltzmann_RCM
from .VAE_net_Gaussian_Boltzmann_UCM import get_network_Gaussian_Boltzmann_UCM
from .VAE_net_Uniform_Boltzmann_RCM import get_network_Uniform_Boltzmann_RCM
from .VAE_net_Uniform_Boltzmann_UCM import get_network_Uniform_Boltzmann_UCM
from .VAE_net_Uniform_Boltzmann_WCM import get_network_Uniform_Boltzmann_WCM
from keras.optimizers import Adam

import os
import gc
import keras
import time
from copy import deepcopy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from random import randint, seed
import numpy as np
from datetime import datetime

import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, Activation, Lambda,Flatten,Dense,Reshape
from keras.layers.merge import Concatenate
from keras.applications import VGG16
from keras import backend as K
from tensorflow.python.ops import math_ops



if tf.__version__ < "2.2.0":
    from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    print('\x1b[6;30;44m' + 'Currently importing callbacks from TensorFlow version:' + '\x1b[0m', tf.__version__)
else:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    print('\x1b[6;30;44m' + 'Currently importing callbacks from TensorFlow version:' + '\x1b[0m', tf.__version__)


import keras
from keras import optimizers, losses


import gc
import warnings


from numpy import inf
from keras  import backend as K
# from keras.preprocessing.image import  ImageDataGenerator
from scipy.spatial.distance import directed_hausdorff, chebyshev
# import horovod.keras as hvd

# force data format to "channels first"
# keras.backend.set_image_data_format('channels_first')

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

class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting settings,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """
    #
    # def __init__(self, filepath, monitor='val_loss', verbose=0,
    #              save_best_only=False, save_weights_only=False,
    #              mode='auto', period=1):
    #     super(ModelCheckpoint, self).__init__()
    #     self.monitor = monitor
    #     self.verbose = verbose
    #     self.filepath = filepath
    #     self.save_best_only = save_best_only
    #     self.save_weights_only = save_weights_only
    #     self.period = period
    #     self.epochs_since_last_save = 0
    #
    #     if mode not in ['auto', 'min', 'max']:
    #         warnings.warn('ModelCheckpoint mode %s is unknown, '
    #                       'fallback to auto mode.' % (mode),
    #                       RuntimeWarning)
    #         mode = 'auto'
    #
    #     if mode == 'min':
    #         self.monitor_op = np.less
    #         self.best = np.Inf
    #     elif mode == 'max':
    #         self.monitor_op = np.greater
    #         self.best = -np.Inf
    #     else:
    #         if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
    #             self.monitor_op = np.greater
    #             self.best = -np.Inf
    #         else:
    #             self.monitor_op = np.less
    #             self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # logs = logs or {}
        # self.epochs_since_last_save += 1
        # if self.epochs_since_last_save >= self.period:
        #     self.epochs_since_last_save = 0
        #     filepath = self.filepath.format(epoch=epoch + 1, **logs)
        #     if self.save_best_only:
        #         current = logs.get(self.monitor)
        #         if current is None:
        #             warnings.warn('Can save best model only with %s available, '
        #                           'skipping.' % (self.monitor), RuntimeWarning)
        #         else:
        #             if self.monitor_op(current, self.best):
        #                 if self.verbose > 0:
        #                     print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
        #                           ' saving model to %s'
        #                           % (epoch + 1, self.monitor, self.best,
        #                              current, filepath))
        #                 self.best = current
        #                 if self.save_weights_only:
        #                     self.model.save_weights(filepath, overwrite=True)
        #                 else:
        #                     self.model.save(filepath, overwrite=True)
        #             else:
        #                 if self.verbose > 0:
        #                     print('\nEpoch %05d: %s did not improve from %0.5f' %
        #                           (epoch + 1, self.monitor, self.best))
        #     else:
        #         if self.verbose > 0:
        #             print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
        #         if self.save_weights_only:
        #             self.model.save_weights(filepath, overwrite=True)
        #         else:
        #             self.model.save(filepath, overwrite=True)
        print(CYELLOW +'Full garbage collection:'+ CEND, 'epoch {}'.format(epoch + 1))
        gc.collect()
        print(gc.get_stats())





class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def PSNR(y_true, y_pred):

    return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

def t_func(arg):
        # arg = np.array(arg)
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
        return arg

def build_and_compile_models(settings, img_rows, img_cols, latent_dim):

    if not os.path.exists(os.path.join(settings['model_saved_paths'],
                                       settings['modelname'])):
        os.mkdir(os.path.join(settings['model_saved_paths'],
                              settings['modelname']))
    if not os.path.exists(os.path.join(settings['model_saved_paths'],
                                       settings['modelname'], 'models')):
        os.mkdir(os.path.join(settings['model_saved_paths'],
                              settings['modelname'], 'models'))
    if settings['debug']:
        if not os.path.exists(os.path.join(settings['model_saved_paths'],
                                           settings['modelname'],
                                           '.train')):
            os.mkdir(os.path.join(settings['model_saved_paths'],
                                  settings['modelname'],
                                  '.train'))
    # settings['URCM'] = (traintest_config.get('nets', 'URCM'))
    # settings['UUCM'] = (traintest_config.get('nets', 'UUCM'))
    # settings['UWCM'] = (traintest_config.get('nets', 'UWCM'))
    # settings['GRCM'] = (traintest_config.get('nets', 'GRCM'))
    # settings['GUCM'] = (traintest_config.get('nets', 'GUCM'))
    # settings['GWCM'] = (traintest_config.get('nets', 'GWCM'))
    # settings['RCON'] = (traintest_config.get('nets', 'RCON'))

    if settings['URCM']:
        model, encoder, decoder = get_network_Uniform_Boltzmann_RCM(img_rows, img_cols, latent_dim)

        def PSNR_URCM(y_true, y_pred):

            a1, b1, z1 = encoder(y_true)
        # z = sampling(z_mean)
            dec1x, dec2x, dec1y, dec2y, dec1z, dec2z = decoder(z1)
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

        # return - 10.0 * K.log(K.mean(K.square(y_pred_right[:,:,:,0] - y_true[:,:,:,0]))) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred_left[:,:,:,0] - y_true[:,:,:,1]))) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(z_pred_left_right[:,:,:,0] - y_true[:,:,:,2]))) / K.log(10.0)

            return - 10.0 * K.log(K.mean(t_func([K.mean(K.square(y_pred_right[:,:,:,0] - y_true[:,:,:,0])), K.mean(K.square(y_pred_left[:,:,:,1] - y_true[:,:,:,1])),  K.mean(K.square(z_pred_left_right[:,:,:,2] - y_true[:,:,:,2]))]))) / K.log(10.0)




        def compute_loss_URCM(y_true, y_pred):


            a1, b1, z1 = encoder(y_true)
            dec1x, dec2x, dec1y, dec2y, dec1z, dec2z = decoder(z1)
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



            bin_loss_right1 = keras.losses.binary_crossentropy(y_true[:,:,:,0], y_pred_right[:,:,:,0])
            bin_loss_right = bin_loss_right1
            bin_loss_left2 = keras.losses.binary_crossentropy(y_true[:,:,:,1], y_pred_left[:,:,:,1])
            bin_loss_left =  bin_loss_left2
            bin_loss_z_pred_left_right3 = keras.losses.binary_crossentropy(y_true[:,:,:,2], z_pred_left_right[:,:,:,2])
            bin_loss_z_pred_left_right =  bin_loss_z_pred_left_right3
            _FLOATX = 'float32'
            dtype = _FLOATX
            seed = np.random.randint(10e6)
            p_z = K.random_uniform(shape=K.shape(z1), minval=1e-8, maxval=1.0,
                               dtype=dtype, seed=seed)
            @tf.function
            def loss_p():
                if tf.reduce_mean(tf.reduce_sum(z1)) > tf.reduce_mean(tf.reduce_sum(a1)) and tf.reduce_mean(tf.reduce_sum(z1)) < tf.reduce_mean(tf.reduce_sum(b1)):
                    if tf.reduce_mean(tf.reduce_sum(b1)) - tf.reduce_mean(tf.reduce_sum(a1)) != 0:
                        # loss = -1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.abs(1.0 + 1e-8 / b1 - a1 + 1e-8))))\
                        #        + 1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(p_z)))
                        return   -1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.maximum(1e-8, tf.math.abs(tf.math.divide_no_nan(1.0,  b1 - a1)))))) \
                                 + 1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(p_z)))

                    else:
                        return   -1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.maximum(1e-8, tf.math.abs(tf.math.divide_no_nan(1.0,  b1)))))) + \
                                 1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(p_z)))


                else:
                    # setting explicitly p_z = 0
                    return  0.0

            loss_px = loss_p()
            return img_rows * img_cols * (bin_loss_right + bin_loss_left + bin_loss_z_pred_left_right) + loss_px



        model.compile(
            optimizer=Adam(lr=1e-4),
            # optimizer=Adam(lr=0.0002),
            # optimizer=Adam(lr=0.00001),
            loss=compute_loss_URCM,
            # loss='binary_crossentropy',
            metrics=[PSNR_URCM]
        )

    if settings['UUCM']:
        model, encoder, decoder = get_network_Uniform_Boltzmann_UCM(img_rows, img_cols, latent_dim)

        def PSNR_UUCM(y_true, y_pred):

            a1, b1, z1 = encoder(y_true)
        # z = sampling(z_mean)
            dec1, dec2 = decoder(z1)
            dec1 = tf.math.divide_no_nan(1.0, dec1)
            dec2 = tf.math.divide_no_nan(1.0, dec2)
            d1d2 = tf.math.multiply(dec1, dec2)
            d1d2n = tf.math.add(1.0, d1d2)
            y_pred = tf.math.divide_no_nan(d1d2, d1d2n)

            return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


        def compute_loss_UUCM(y_true, y_pred):

            a1, b1, z1 = encoder(y_true)
        # z = sampling(z_mean)
            dec1, dec2 = decoder(z1)
            dec1 = tf.math.divide_no_nan(1.0, dec1)
            dec2 = tf.math.divide_no_nan(1.0, dec2)
            d1d2 = tf.math.multiply(dec1 , dec2)
            d1d2n = tf.math.add(1.0, d1d2)
            y_pred = tf.math.divide_no_nan(d1d2, d1d2n)
            bin_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        # loss1 = 0.0
        # loss2 = 0.0
            _FLOATX = 'float32'
            dtype = _FLOATX
            seed = np.random.randint(10e6)
            p_z = K.random_uniform(shape=K.shape(z1), minval=1e-8, maxval=1.0,
                               dtype=dtype, seed=seed)
            @tf.function
            def loss_p():
                if tf.reduce_mean(tf.reduce_sum(z1)) > tf.reduce_mean(tf.reduce_sum(a1)) and tf.reduce_mean(tf.reduce_sum(z1)) < tf.reduce_mean(tf.reduce_sum(b1)):
                    if tf.reduce_mean(tf.reduce_sum(b1)) - tf.reduce_mean(tf.reduce_sum(a1)) != 0:
                        # loss = -1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.abs(1.0 + 1e-8 / b1 - a1 + 1e-8))))\
                        #        + 1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(p_z)))
                        return   -1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.maximum(1e-8, tf.math.abs(tf.math.divide_no_nan(1.0,  b1 - a1)))))) \
                                 + 1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(p_z)))

                    else:
                        return   -1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.maximum(1e-8, tf.math.abs(tf.math.divide_no_nan(1.0,  b1)))))) + \
                                 1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(p_z)))


                else:
                    # setting explicitly p_z = 0
                    return  0.0

            loss_px = loss_p()

            return loss_px + img_rows * img_cols * 3 * bin_loss


        model.compile(
            optimizer=Adam(lr=1e-4),
            # optimizer=Adam(lr=0.0002),
            # optimizer=Adam(lr=0.00001),
            loss=compute_loss_UUCM,
            # loss='binary_crossentropy',
            metrics=[PSNR_UUCM]
        )

    if settings['UWCM']:
        model, encoder, decoder = get_network_Uniform_Boltzmann_WCM(img_rows, img_cols, latent_dim)

        def PSNR_UWCM(y_true, y_pred):

            a1, b1, z1 = encoder(y_true)
        # z = sampling(z_mean)


            dec1, dec2 = decoder(z1)
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
            ypred = tf.math.divide_no_nan(dec1x_dec2y, d1d2_top)
        # ypred = tf.clip_by_value(ypred_x, 0.0, 1.0)


            return - 10.0 * K.log(K.mean(K.square(ypred - y_true))) / K.log(10.0)

        def compute_loss_UWCM(y_true, y_pred):

            a1, b1, z1 = encoder(y_true)
        # z = sampling(z_mean)
            dec1, dec2 = decoder(z1)
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
            ypred = tf.math.divide_no_nan(dec1x_dec2y, d1d2_top)


            lossbin = keras.losses.binary_crossentropy(y_true, ypred)

            _FLOATX = 'float32'
            dtype = _FLOATX
            seed = np.random.randint(10e6)
            p_z = K.random_uniform(shape=K.shape(z1), minval=1e-8, maxval=1.0,
                               dtype=dtype, seed=seed)
            @tf.function
            def loss_p():
                if tf.reduce_mean(tf.reduce_sum(z1)) > tf.reduce_mean(tf.reduce_sum(a1)) and tf.reduce_mean(tf.reduce_sum(z1)) < tf.reduce_mean(tf.reduce_sum(b1)):
                    if tf.reduce_mean(tf.reduce_sum(b1)) - tf.reduce_mean(tf.reduce_sum(a1)) != 0:
                        # loss = -1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.abs(1.0 + 1e-8 / b1 - a1 + 1e-8))))\
                        #        + 1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(p_z)))
                        return   -1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.maximum(1e-8, tf.math.abs(tf.math.divide_no_nan(1.0,  b1 - a1)))))) \
                                 + 1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(p_z)))

                    else:
                        return   -1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.maximum(1e-8, tf.math.abs(tf.math.divide_no_nan(1.0,  b1)))))) + \
                                 1.0 * tf.reduce_mean(tf.reduce_sum(tf.math.log(p_z)))


                else:
                    # setting explicitly p_z = 0
                    return  0.0

            loss_px = loss_p()


            return img_rows * img_cols * 3 * lossbin + loss_px


        model.compile(
            optimizer=Adam(lr=1e-4),
            # optimizer=Adam(lr=0.0002),
            # optimizer=Adam(lr=0.00001),
            loss=compute_loss_UWCM,
            # loss='binary_crossentropy',
            metrics=[PSNR_UWCM]
        )

    if settings['GRCM']:
        model, encoder, decoder = get_network_Gaussian_Boltzmann_RCM(img_rows, img_cols, latent_dim)

        def PSNR_GRCM(y_true, y_pred):

            y_pred = tf.convert_to_tensor(y_pred)

            y_true = tf.cast(y_true, y_pred.dtype)


            y_error = y_true - y_true
            mask = math_ops.abs(1 - y_error)


        # masked = tf.math.multiply(y_true, tmp)
            masked = tf.math.multiply(y_true, mask)
            a1, b1, z1 = encoder([masked, mask])
        # z = sampling(z_mean)decoder([inputs_img, inputs_mask,encoded_img3])

        # dec1, dec2 = decoder([masked, mask, z1])

            dec1x, dec2x, dec1y, dec2y, dec1z, dec2z = decoder([masked, mask, z1])
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

            tmp = K.mean(t_func([K.mean(K.square(y_pred_right[:, :, :, 0] - y_true[:, :, :, 0])), K.mean(K.square(y_pred_left[:, :, :, 1] - y_true[:, :, :, 1])), K.mean(K.square(z_pred_left_right[:, :, :, 2] - y_true[:, :, :, 2]))]))

            # if tmp <= 0:
            #       return - 10.0 * tf.math.log(1e-8) / K.log(10.0)
            # else:
            return - 10.0 * tf.math.log(tmp) / K.log(10.0)

        def compute_loss_GRCM(y_true, y_pred):

            y_pred = tf.convert_to_tensor(y_pred)

            y_true = tf.cast(y_true, y_pred.dtype)

            #
            y_error = y_true - y_true
            mask = math_ops.abs(1 - y_error)

            masked = tf.math.multiply(y_true, mask)
            a1, b1, z1 = encoder([masked, mask])

            kl_loss = 1 + b1 - K.square(a1) - K.exp(b1)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            kl_loss = K.mean(kl_loss)
            dec1x, dec2x, dec1y, dec2y, dec1z, dec2z = decoder([masked, mask, z1])
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



            bin_loss_right = keras.losses.binary_crossentropy(y_true[:,:,:,0], y_pred_right[:,:,:,0])
            bin_loss_left = keras.losses.binary_crossentropy(y_true[:,:,:,1], y_pred_left[:,:,:,1])
            bin_loss_z_pred_left_right = keras.losses.binary_crossentropy(y_true[:,:,:,2], z_pred_left_right[:,:,:,2])
            h = tf.keras.losses.Huber()


            c1= h(y_true[:,:,:,0], y_pred_right[:,:,:,0]) + tf.keras.losses.mae(y_true[:,:,:,0], y_pred_right[:,:,:,0]) + tf.keras.losses.MAE(y_true[:,:,:,0], y_pred_right[:,:,:,0])
            c2= h(y_true[:,:,:,1], y_pred_left[:,:,:,1]) + tf.keras.losses.mae(y_true[:,:,:,1], y_pred_left[:,:,:,1]) + tf.keras.losses.MAE(y_true[:,:,:,1], y_pred_left[:,:,:,1])
            c3= h(y_true[:,:,:,2], z_pred_left_right[:,:,:,2]) + tf.keras.losses.mae(y_true[:,:,:,2], z_pred_left_right[:,:,:,2]) + tf.keras.losses.MAE(y_true[:,:,:,2], z_pred_left_right[:,:,:,2])

            return img_rows * img_cols * (bin_loss_right + bin_loss_left + bin_loss_z_pred_left_right + c1 + c2 + c3) + kl_loss

        model.compile(
            optimizer=Adam(lr=0.0002, clipvalue=0.5, clipnorm=0.5),
            # optimizer=Adam(lr=0.0002),
            # optimizer=Adam(lr=0.00001),
            loss=compute_loss_GRCM,
            # loss='binary_crossentropy',
            metrics=[PSNR_GRCM]
        )
    if settings['GUCM']:
        model, encoder, decoder = get_network_Gaussian_Boltzmann_UCM(img_rows, img_cols, latent_dim)

        def PSNR_GUCM(y_true, y_pred):

            y_pred = tf.convert_to_tensor(y_pred)

            y_true = tf.cast(y_true, y_pred.dtype)


            y_error = y_true - y_true
            mask = math_ops.abs(1 - y_error)
            masked = tf.math.multiply(y_true, mask)
            a1, b1, z1 = encoder([masked, mask])
        # z = sampling(z_mean)decoder([inputs_img, inputs_mask,encoded_img3])

            dec1, dec2 = decoder([masked, mask, z1])
            dec1 = tf.math.divide_no_nan(1.0, dec1)
            dec2 = tf.math.divide_no_nan(1.0, dec2)
            d1d2 = tf.math.multiply(dec1, dec2)
            d1d2n = tf.math.add(1.0, d1d2)
            y_pred_ch = tf.math.divide_no_nan(d1d2, d1d2n)

        # return - 10.0 * K.log(K.maximum(1e-8, K.mean(K.square(y_pred_ch - y_true)))) / K.log(10.0)
            tmp = tf.math.maximum(1e-8, tf.math.reduce_mean(tf.math.square(y_pred_ch - y_true)))
            if tmp <= 0:
                return - 10.0 * tf.math.log(1e-8) / K.log(10.0)
            else:
                return - 10.0 * tf.math.log(tmp) / K.log(10.0)

        def compute_loss_GUCM(y_true, y_pred):


            y_pred = tf.convert_to_tensor(y_pred)

            y_true = tf.cast(y_true, y_pred.dtype)

    #
            y_error = y_true - y_true
            mask = math_ops.abs(1 - y_error)


    # masked = tf.math.multiply(y_true, tmp)
            masked = tf.math.multiply(y_true, mask)
            a1, b1, z1 = encoder([masked, mask])
            kl_loss = 1 + b1 - K.square(a1) - K.exp(b1)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            kl_loss = K.mean(kl_loss)
    # z = sampling(z_mean)
            dec1, dec2 = decoder([masked, mask, z1])
            dec1 = tf.math.divide_no_nan(1.0, dec1)
            dec2 = tf.math.divide_no_nan(1.0, dec2)
            d1d2 = tf.math.multiply(dec1, dec2)
            d1d2n = tf.math.add(1.0, d1d2)
            y_pred_ch = tf.math.divide_no_nan(d1d2, d1d2n)
            bin_loss = keras.losses.binary_crossentropy(y_true, y_pred_ch)

            return kl_loss + img_rows * img_cols * bin_loss




        model.compile(
            optimizer=Adam(lr=0.0002, clipvalue=0.5, clipnorm=0.5),
            # optimizer=Adam(lr=0.0002),
            # optimizer=Adam(lr=0.00001),
            loss=compute_loss_GUCM,
            # loss='binary_crossentropy',
            metrics=[PSNR_GUCM]
        )
    if settings['GWCM']:

        model, encoder, decoder = get_network_Gaussian_Boltzmann_WCM(img_rows, img_cols, latent_dim)

        def PSNR_GWCM(y_true, y_pred):
            y_pred = tf.convert_to_tensor(y_pred)

            y_true = tf.cast(y_true, y_pred.dtype)


            y_error = y_true - y_true
            mask = math_ops.abs(1 - y_error)

            masked = tf.math.multiply(y_true, mask)
            a1, b1, z1 = encoder([masked, mask])
    # z = sampling(z_mean)decoder([inputs_img, inputs_mask,encoded_img3])

            dec1, dec2 = decoder([masked, mask, z1])
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
            y_pred_ch = tf.math.divide_no_nan(dec1x_dec2y, d1d2_top)
            tmp = tf.math.maximum(1e-8, tf.math.reduce_mean(tf.math.square(y_pred_ch - y_true)))
            if tmp <= 0:
                 return - 10.0 * tf.math.log(1e-8) / K.log(10.0)
            else:
                return - 10.0 * tf.math.log(tmp) / K.log(10.0)

        def compute_loss_GWCM(y_true, y_pred):

            y_pred = tf.convert_to_tensor(y_pred)

            y_true = tf.cast(y_true, y_pred.dtype)

    #
            y_error = y_true - y_true
            mask = math_ops.abs(1 - y_error)

            masked = tf.math.multiply(y_true, mask)
            a1, b1, z1 = encoder([masked, mask])
            kl_loss = 1 + b1 - K.square(a1) - K.exp(b1)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            kl_loss = K.mean(kl_loss)
    # z = sampling(z_mean)
            dec1, dec2 = decoder([masked, mask, z1])
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
            y_pred_ch = tf.math.divide_no_nan(dec1x_dec2y, d1d2_top)

            bin_loss = keras.losses.binary_crossentropy(y_true, y_pred_ch)
            return kl_loss + img_rows * img_cols * bin_loss




        model.compile(
            optimizer=Adam(lr=0.0002, clipvalue=0.5, clipnorm=0.5),
            # optimizer=Adam(lr=0.0002),
            # optimizer=Adam(lr=0.00001),
            loss=compute_loss_GWCM,
            # loss='binary_crossentropy',
            metrics=[PSNR_GWCM]
        )
    if settings['RCON']:
        model, encoder, decoder = get_network_Gaussian_reconstruction(img_rows, img_cols, latent_dim)

        def PSNR_RCON(y_true, y_pred):
            return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

        def compute_loss_RCON(y_true, y_pred):

            h = tf.keras.losses.Huber()
            Huber = h(y_true, y_pred)
            # '''''''''''''''''''''''''
            def __int_shape(x, back):
                return K.int_shape(x) if back == 'tensorflow' else K.shape(x)
            def _preprocess_padding(padding):
                if padding == 'same':
                    padding = 'SAME'
                elif padding == 'valid':
                    padding = 'VALID'
                else:
                    raise ValueError('Invalid padding:', padding)

                return padding

            def extract_image_patches(x, ksizes, ssizes, padding='same',data_format='channels_last'):

                kernel = [1, ksizes[0], ksizes[1], 1]
                strides = [1, ssizes[0], ssizes[1], 1]
                padding = _preprocess_padding(padding)
                if data_format == 'channels_first':
                    x = K.permute_dimensions(x, (0, 2, 3, 1))
                bs_i, w_i, h_i, ch_i = K.int_shape(x)
                # patches = tf.image.extract_patches(x, kernel, strides, [1, 1, 1, 1], padding)
                patches = tf.image.extract_patches(images=x,
                                                   sizes=kernel,
                                                   strides=strides,
                                                   rates=[1, 1, 1, 1],
                                                   padding=padding)
                bs, w, h, ch = K.int_shape(patches)
                reshaped = tf.reshape(patches, [-1, w, h, tf.math.floordiv(ch, ch_i), ch_i])
                # tf.math.floordiv(
                #     x, y, name=None
                # )
                final_shape = [-1, w, h, ch_i, ksizes[0], ksizes[1]]
                patches = tf.reshape(tf.transpose(reshaped, [0, 1, 2, 4, 3]), final_shape)
                if data_format == 'channels_last':
                    patches = K.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
                return patches

            kernel_size = 11
            max_value = 1.0
            k1 = 0.01
            k2 = 0.03

            c1 = (k1 * max_value) ** 2
            c2 = (k2 * max_value) ** 2
            dim_ordering = K.image_data_format()
            backend = K.backend()
            kernel = [kernel_size, kernel_size]
            y_true_s = K.reshape(y_true, [-1] + list(__int_shape(y_pred, backend)[1:]))
            y_pred_s = K.reshape(y_pred, [-1] + list(__int_shape(y_pred, backend)[1:]))

            patches_pred = extract_image_patches(y_pred_s, kernel, kernel, 'valid',
                                                 dim_ordering)

            patches_true = extract_image_patches(y_true_s, kernel, kernel, 'valid',
                                                 dim_ordering)

            # Reshape to get the var in the cells
            bs, w, h, c1, c2, c3 = __int_shape(patches_pred, backend)
            patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
            patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
            # Get mean
            u_true = K.mean(patches_true, axis=-1)
            u_pred = K.mean(patches_pred, axis=-1)
            # Get variance
            var_true = K.var(patches_true, axis=-1)
            var_pred = K.var(patches_pred, axis=-1)
            # Get std dev
            covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

            ssim = (2 * u_true * u_pred + c1) * (2 * covar_true_pred + c2)
            denom = ((K.square(u_true) + K.square(u_pred) + c1) * (var_pred + var_true + c2))
            ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
            ssim2_loss = K.mean((1.0 - ssim) / 2.0)

            neg_y_true = 1 - y_true
            neg_y_pred = 1 - y_pred

            tp = K.sum(y_true * y_pred)
            fn = K.sum(y_true * neg_y_pred)


            fp = K.sum(neg_y_true * y_pred)
            tn = K.sum(neg_y_true * neg_y_pred)
            mult1 = tp * tn
            mult2 = fp * fn
            sum1 = fp + tp
            sum2 = tp + fn
            sum3 = tn + fp
            sum4 = tn + fn

            mcc = (mult1 - mult2) / tf.math.sqrt(sum1 * sum2 * sum3 * sum4)

            mat_loos = K.abs(1 - mcc)
            smooth = 1.
            # Flatten
            y_true_fx = tf.reshape(y_true, [-1])
            y_pred_fx = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_fx * y_pred_fx)
            score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_fx) + tf.reduce_sum(y_pred_fx) + smooth)
            score = 1 - score



            y_pred = tf.convert_to_tensor(y_pred)

            y_true = tf.cast(y_true, y_pred.dtype)


            y_error = y_pred - y_true

            y_comp = tf.math.multiply(y_error, y_true) + tf.math.multiply((1-y_error),  y_pred)


            l1_loss = tf.reduce_sum(math_ops.abs(y_pred - y_true), axis=-1)
            # l1_loss_sum = tf.reduce_sum(math_ops.abs(y_pred - y_true), axis=-1)
            l1_lossx = tf.reduce_sum(math_ops.abs(1 - y_error), axis=-1)
            l1_lossx = 1 - l1_lossx


            l1_loss_1 = tf.reduce_sum(math_ops.abs(y_pred[:,:,:,0] - y_true[:,:,:,0]))
            l1_loss_2 = tf.reduce_sum(math_ops.abs(y_pred[:,:,:,1] - y_true[:,:,:,1]))
            l1_loss_3 = tf.reduce_sum(math_ops.abs(y_pred[:,:,:,2] - y_true[:,:,:,2]))
            # mask = tf.math.multiply(y_error, y_true)
            mask = math_ops.abs(y_error)
            tmp = math_ops.abs(1 - y_error)

            # masked = tf.math.multiply(y_true, tmp)
            masked = tf.math.multiply(y_true, tmp)
            z_mean, z_log_var, z = encoder([masked, mask])
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            kl_loss = K.mean(kl_loss)

            return l1_loss + l1_lossx + l1_loss_1 + l1_loss_2 + l1_loss_3 + ssim2_loss + Huber + kl_loss

        model.compile(
            optimizer=Adam(lr=0.0002, clipvalue=0.5, clipnorm=0.5),
            # optimizer=Adam(lr=0.0002),
            # optimizer=Adam(lr=0.00001),
            loss=compute_loss_RCON,
            # loss='binary_crossentropy',
            metrics=[PSNR_RCON]
        )

    try:
        model = tf.keras.utils.multi_gpu_model(model, cpu_relocation=True)
        print("Training model using multiple GPUs..")
    except:
        print("Training model using single GPU or CPU..")

    net_model_1 = 'model_1'
    net_weights_1 = os.path.join(settings['model_saved_paths'],
                                settings['modelname'],
                                'models', net_model_1 + '.hdf5')

    network1 = {}
    network1['model'] = model
    network1['weights'] = net_weights_1
    network1['history'] = None
    network1['special_name_1'] = net_model_1



    if settings['load_weights'] is True:
        print("> CNN: loading weights from", settings['modelname'], 'configuration')
        print(net_weights_1)

        network1['model'].load_weights(net_weights_1, by_name=True)


    return network1


THIS_PATH = os.path.split(os.path.realpath(__file__))[0]
Image_save = THIS_PATH + '/temporary_image_outputs/img_{}.png'


def image_out_epoch(model, test_generator, settings):

    if settings['GWCM']:

        print('\x1b[6;30;41m' + "Training Weighted configuration model (WCM)-Gaussian" + '\x1b[0m')

        test_data = next(test_generator)
        (masked, mask), ori = test_data
        dec1, dec2 = model.predict([masked, mask])
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
            plt.savefig(Image_save.format(i))
            plt.close()

    if settings['GUCM']:
        print('\x1b[6;30;41m' + "Training Undirected configuration model (UCM)-Gaussian" + '\x1b[0m')
        test_data = next(test_generator)
        (masked, mask), ori = test_data
        pred_img1, pred_img2 = model.predict([masked, mask])
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
            plt.savefig(Image_save.format(i))
            plt.close()

    if settings['GRCM']:
        print('\x1b[6;30;41m' + "Training Reciprocal configuration model (RCM)-Gaussian" + '\x1b[0m')
        test_data = next(test_generator)
        (masked, mask), ori = test_data
        dec1x, dec2x, dec1y, dec2y, dec1z, dec2z = model.predict([masked, mask])

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
            plt.savefig(Image_save.format(i))
            plt.close()

    if settings['UWCM']:
        print('\x1b[6;30;41m' + "Training Weighted configuration model (WCM)-Uniform" + '\x1b[0m')
        test_data = next(test_generator)
        (ori, mask) = test_data
        dec1, dec2 = model.predict(ori)

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
            plt.savefig(Image_save.format(i))
            plt.close()

    if settings['UUCM']:
        print('\x1b[6;30;41m' + "Training Undirected configuration model (UCM)-Uniform" + '\x1b[0m')
        test_data = next(test_generator)
        (ori, mask) = test_data
        pred_img1, pred_img2 = model.predict(ori)
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
            plt.savefig(Image_save.format(i))
            plt.close()

    if settings['URCM']:
        print('\x1b[6;30;41m' + "Training Reciprocal configuration model (RCM)-Uniform" + '\x1b[0m')
        test_data = next(test_generator)
        (ori, mask) = test_data
        dec1x, dec2x, dec1y, dec2y, dec1z, dec2z = model.predict(ori)

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
            plt.savefig(Image_save.format(i))
            plt.close()

    if settings['RCON']:
        print('\x1b[6;30;41m' + "Training reconstruction model (RCON)-Gaussian" + '\x1b[0m')
        test_data = next(test_generator)
        (masked, mask), ori = test_data
        pred_img7 = model.predict([masked, mask])

        # Clear current output and display test_done images
        for i in range(len(ori)):
            _, axes = plt.subplots(1, 2, figsize=(20, 5))
            axes[0].imshow(masked[i,:,:,:])
            axes[1].imshow(pred_img7[i,:,:,:])
            axes[0].set_title('Original Image')
            axes[1].set_title('Predicted Image')
            plt.savefig(Image_save.format(i))
            plt.close()


def train_model_all(train_model, train_generator, val_generator, test_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID, settings):


    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=20, verbose=1,
        mode='auto', min_delta=0.001, cooldown=0, min_lr=0)
    if test_generator is not None:
        train_model['model'].fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VALID,
            # steps_per_epoch=1,
            # validation_data=val_generator,
            # validation_steps=1,
            epochs=2000,
            verbose=1,
            callbacks=[reduce_lr, TensorBoard(log_dir='./tensorboardlogs', histogram_freq=0,
                                              write_graph=True,  write_images=True),
                       ModelCheckpoint(
                           train_model['weights'],
                           monitor='val_loss',
                           save_best_only=True,
                           save_weights_only=True
                       ),
                       LambdaCallback(
                           on_epoch_end=lambda epoch, logs: image_out_epoch(train_model['model'], test_generator, settings)
                       )
                       ])

    else:
        train_model['model'].fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VALID,
            # steps_per_epoch=1,
            # validation_data=val_generator,
            # validation_steps=1,
            epochs=2000,
            verbose=1,
            callbacks=[reduce_lr, TensorBoard(log_dir='./tensorboardlogs', histogram_freq=0,
                                              write_graph=True,  write_images=True),
                       ModelCheckpoint(
                           train_model['weights'],
                           monitor='val_loss',
                           save_best_only=True,
                           save_weights_only=True
                       )
                       ])

    return train_model

