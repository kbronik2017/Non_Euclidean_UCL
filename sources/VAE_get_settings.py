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

def TrainandTest_settings(traintest_config):

    settings = {}
    settings['modelname'] = traintest_config.get('traintestset', 'name')
    settings['training_folder'] = traintest_config.get('traintestset', 'training_folder')
    settings['cross_validation_folder'] = traintest_config.get('traintestset', 'cross_validation_folder')
    settings['inference_folder'] = traintest_config.get('traintestset', 'inference_folder')
    settings['debug'] = traintest_config.get('traintestset', 'debug')
    settings['save_tmp'] = (traintest_config.get('traintestset', 'save_tmp'))
    settings['URCM'] = (traintest_config.get('nets', 'URCM'))
    settings['UUCM'] = (traintest_config.get('nets', 'UUCM'))
    settings['UWCM'] = (traintest_config.get('nets', 'UWCM'))
    settings['GRCM'] = (traintest_config.get('nets', 'GRCM'))
    settings['GUCM'] = (traintest_config.get('nets', 'GUCM'))
    settings['GWCM'] = (traintest_config.get('nets', 'GWCM'))
    settings['RCON'] = (traintest_config.get('nets', 'RCON'))
    settings['gpu_number'] = traintest_config.getint('traintestset', 'gpu_number')
    settings['learnedmodel'] = traintest_config.get('traintestset', 'learnedmodel')
    settings['model_saved_paths'] = None
    settings['max_epochs'] = traintest_config.getint('traintestset', 'max_epochs')
    settings['patience'] = traintest_config.getint('traintestset', 'patience')
    settings['batch_size'] = traintest_config.getint('traintestset', 'batch_size')
    settings['net_verbose'] = traintest_config.getint('traintestset', 'net_verbose')
    settings['tensorboard'] = traintest_config.get('tensorboard', 'tensorboard_folder')
    settings['port'] = traintest_config.getint('tensorboard', 'port')
    settings['load_weights'] = True
    settings['randomize_train'] = True
    settings['error_tolerance'] = traintest_config.getfloat('traintestset',
                                                   'error_tolerance')
    settings['full_train'] = (traintest_config.get('traintestset', 'full_train'))

    settings['learnedmodel_model'] = traintest_config.get('traintestset',
                                                     'learnedmodel_model')
    settings['num_layers'] = None

    keys = list(settings.keys())
    for k in keys:
        value = settings[k]
        if value == 'True':
            settings[k] = True
        if value == 'False':
            settings[k] = False

    return settings

def Train_Test_settings_show(settings):
    print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
    print('\x1b[6;30;45m' + 'Train/Test settings' + '\x1b[0m')
    print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
    print(" ")
    keys = list(settings.keys())
    for key in keys:
        print(CRED + key, ':' + CEND, settings[key])
    print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')