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
import time
import numpy as np
from nibabel import load as load_nii
import nibabel as nib
from sklearn import preprocessing
from operator import itemgetter
from sources.VAE_model import train_model_all
from operator import add
from keras.models import load_model
import tensorflow as tf
import configparser

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

def train_vae_model(train_model, train_generator, val_generator, test_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID, settings, thispath):

    # ----------
    # CNN1
    # ----------
    traintest_config = configparser.ConfigParser()
    traintest_config.read(os.path.join(thispath, 'config', 'configuration.cfg'))
    print("Variational Auto Encoder (VAE) loading training data for:")

    if settings['GWCM']:
        print('\x1b[6;30;41m' + "Training Weighted configuration model (WCM)-Gaussian" + '\x1b[0m')
    if settings['GUCM']:
        print('\x1b[6;30;41m' + "Training Undirected configuration model (UCM)-Gaussian" + '\x1b[0m')
    if settings['GRCM']:
        print('\x1b[6;30;41m' + "Training Reciprocal configuration model (RCM)-Gaussian" + '\x1b[0m')
    if settings['UWCM']:
        print('\x1b[6;30;41m' + "Training Weighted configuration model (WCM)-Uniform" + '\x1b[0m')
    if settings['UUCM']:
        print('\x1b[6;30;41m' + "Training Undirected configuration model (UCM)-Uniform" + '\x1b[0m')
    if settings['URCM']:
        print('\x1b[6;30;41m' + "Training Reciprocal configuration model (RCM)-Uniform" + '\x1b[0m')
    if settings['RCON']:
        print('\x1b[6;30;41m' + "Training reconstruction model (RCON)-Gaussian" + '\x1b[0m')

    print("")

    print('\x1b[6;30;41m' + "                                                               " + '\x1b[0m')
    print('\x1b[6;30;41m' + "Loading data into memory..., training will begin shortly ...   " + '\x1b[0m')
    print('\x1b[6;30;41m' + "                                                               " + '\x1b[0m')
    net_model_name = train_model['special_name_1']
    if os.path.exists(os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name + '.hdf5')):
        net_weights_1 = os.path.join(settings['model_saved_paths'], settings['modelname'],'models', net_model_name + '.hdf5')
        try:

            train_model['model'].load_weights(net_weights_1, by_name=True)
            print("CNN has Loaded previous weights from the", net_weights_1)
        except:
            print("> ERROR: The model", settings['modelname'],'selected does not contain a valid network model')
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)
    else:
        pass

    trained_model = train_model_all(train_model, train_generator, val_generator, test_generator, STEP_SIZE_TRAIN, STEP_SIZE_VALID, settings)


    return trained_model








