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

from timeit import time
import configparser
import argparse
import platform
import subprocess
import os
import signal
import queue
import threading
from __init__ import __version__
from tkinter import Frame, LabelFrame, Label, END, Tk
from tkinter import Entry, Button, Checkbutton, OptionMenu, Toplevel, Text
from tkinter import BooleanVar, StringVar, IntVar, DoubleVar
from tkinter.filedialog import askdirectory
from tkinter.ttk import Notebook
# from tkinter import *
from PIL import Image, ImageTk
import webbrowser

from VAE_cnn_main import train_network_vae, train_test_network_vae, infer_vae, overall_config
##################

from tkinter.ttk import Label
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

class AnimatedGIF(Label, object):
    def __init__(self, master, path, forever=True):
        self._master = master
        self._loc = 0
        self._forever = forever

        self._is_running = False

        im = Image.open(path)
        self._frames = []
        i = 0
        try:
            while True:
                # photoframe = ImageTk.PhotoImage(im.copy().convert('RGBA'))
                photoframe = ImageTk.PhotoImage(im)
                self._frames.append(photoframe)

                i += 1
                im.seek(i)
        except EOFError:
            pass

        self._last_index = len(self._frames) - 1

        try:
            self._delay = im.info['duration']
        except:
            self._delay = 1000

        self._callback_id = None

        super(AnimatedGIF, self).__init__(master, image=self._frames[0])

    def start_animation(self, frame=None):
        if self._is_running: return

        if frame is not None:
            self._loc = 0
            self.configure(image=self._frames[frame])

        self._master.after(self._delay, self._animate_GIF)
        self._is_running = True

    def stop_animation(self):
        if not self._is_running: return

        if self._callback_id is not None:
            self.after_cancel(self._callback_id)
            self._callback_id = None

        self._is_running = False

    def _animate_GIF(self):
        self._loc += 1
        self.configure(image=self._frames[self._loc])

        if self._loc == self._last_index:
            if self._forever:
                self._loc = 0
                self._callback_id = self._master.after(self._delay, self._animate_GIF)
            else:
                self._callback_id = None
                self._is_running = False
        else:
            self._callback_id = self._master.after(self._delay, self._animate_GIF)

    def pack(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(AnimatedGIF, self).pack(**kwargs)

    def grid(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(AnimatedGIF, self).grid(**kwargs)

    def place(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(AnimatedGIF, self).place(**kwargs)

    def pack_forget(self, **kwargs):
        self.stop_animation()

        super(AnimatedGIF, self).pack_forget(**kwargs)

    def grid_forget(self, **kwargs):
        self.stop_animation()

        super(AnimatedGIF, self).grid_forget(**kwargs)

    def place_forget(self, **kwargs):
        self.stop_animation()

        super(AnimatedGIF, self).place_forget(**kwargs)


class CNN_VAE:

    def __init__(self, master, container):

        self.master = master
        master.title("UCL Engineering 2021")


        self.path = os.getcwd()
        self.traintest_config = None
        self.user_config = None
        self.current_folder = os.getcwd()
        self.list_train_learnedmodel_nets = []

        self.list_test_nets = []
        self.version = __version__
        self.training_do = None
        self.testing_do = None
        self.test_queue = queue.Queue()
        self.train_queue = queue.Queue()
        self.setting_training_folder = StringVar()
        self.setting_gen_tag = StringVar()
        self.setting_cross_validation = StringVar()
        self.setting_test_folder = StringVar()
        self.setting_tensorboard_folder = StringVar()
        self.setting_port_value = IntVar()
        self.setting_PORT_tag = IntVar()
        self.setting_save_tmp = BooleanVar()
        self.setting_debug = BooleanVar()
        self.setting_net_folder = os.path.join(self.current_folder, 'models')
        self.setting_use_learnedmodel_model = BooleanVar()
        self.setting_learnedmodel_model = StringVar()
        self.setting_inference_model = StringVar()
        self.setting_num_layers = IntVar()
        self.setting_net_name = StringVar()
        self.setting_net_name.set('None')
        self.setting_vae = BooleanVar()
        self.URCM_train = BooleanVar()
        self.UUCM_train = BooleanVar()
        self.UWCM_train = BooleanVar()
        self.GRCM_train = BooleanVar()
        self.GUCM_train = BooleanVar()
        self.GWCM_train = BooleanVar()
        self.RCON_train = BooleanVar()
        self.pre_processing = BooleanVar()
        self.setting_learnedmodel = None
        self.setting_min_th = DoubleVar()
        self.setting_patch_size = IntVar()
        self.setting_weight_paths = StringVar()
        self.setting_load_weights = BooleanVar()
        self.setting_max_epochs = IntVar()
        self.setting_patience = IntVar()
        self.setting_batch_size = IntVar()
        self.setting_net_verbose = IntVar()
        self.setting_threshold = DoubleVar()
        self.setting_error_tolerance = DoubleVar()
        self.setting_mode = BooleanVar()
        self.setting_gpu_number = IntVar()
        self.load_traintest_configuration()
        self.updated_traintest_configuration()
        self.note = Notebook(self.master)
        self.note.pack()
        os.system('cls' if platform.system() == 'Windows' else 'clear')

        print("##################################################")
        print('\x1b[6;29;45m' + 'Variational Auto Encoder (VAE)              ' + '\x1b[0m')
        print('\x1b[6;29;45m' + 'Gaussian Uniform Boltzmann                  ' + '\x1b[0m')
        print('\x1b[6;29;45m' + 'UCL Engineering-Computer Science Department ' + '\x1b[0m')
        print('\x1b[6;29;45m' + 'Kevin Bronik (2020-2021)                    ' + '\x1b[0m')
        print("##################################################")

        self.train_frame = Frame()
        self.note.add(self.train_frame, text="Training and Inference")

        # label frames
        cl_s = 6
        self.tr_frame = LabelFrame(self.train_frame, text="Training images:")
        self.tr_frame.grid(row=0, columnspan=cl_s, sticky='WE',
                           padx=5, pady=5, ipadx=5, ipady=5)
        self.model_frame = LabelFrame(self.train_frame, text="CNN model:")
        self.model_frame.grid(row=5, columnspan=cl_s, sticky='WE',
                              padx=5, pady=5, ipadx=5, ipady=5)

        # training settings
        self.inFolderLbl = Label(self.tr_frame, text="Training folder:")
        self.inFolderLbl.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        self.inFolderTxt = Entry(self.tr_frame)
        self.inFolderTxt.grid(row=0,
                              column=1,
                              columnspan=5,
                              sticky="W",
                              pady=3)
        self.inFileBtn = Button(self.tr_frame, text="Browse ...",
                                command=self.load_training_path)
        self.inFileBtn.grid(row=0,
                            column=5,
                            columnspan=1,
                            sticky='W',
                            padx=5,
                            pady=1)

        self.incvFolderLbl = Label(self.tr_frame, text="Cross-validation folder:")
        self.incvFolderLbl.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        self.incvFolderTxt = Entry(self.tr_frame)
        self.incvFolderTxt.grid(row=1,
                              column=1,
                              columnspan=5,
                              sticky="W",
                              pady=3)
        self.incvFileBtn = Button(self.tr_frame, text="Browse ...",
                                command=self.load_cross_validation_path)
        self.incvFileBtn.grid(row=1,
                            column=5,
                            columnspan=1,
                            sticky='W',
                            padx=5,
                            pady=1)


        self.test_inFolderLbl = Label(self.tr_frame, text="Inference folder:")
        self.test_inFolderLbl.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        self.test_inFolderTxt = Entry(self.tr_frame)
        self.test_inFolderTxt.grid(row=2,
                                   column=1,
                                   columnspan=5,
                                   sticky="W",
                                   pady=3)
        self.test_inFileBtn = Button(self.tr_frame, text="Browse ...",
                                     command=self.load_testing_path)
        self.test_inFileBtn.grid(row=2,
                                 column=5,
                                 columnspan=1,
                                 sticky='W',
                                 padx=5,
                                 pady=1)



        self.settingsBtn = Button(self.tr_frame,
                                 text="Settings",
                                 command=self.settings)
        self.settingsBtn.grid(row=0,
                             column=10,
                             columnspan=1,
                             sticky="W",
                             padx=(30, 1),
                             pady=1)

        self.train_aboutBtn = Button(self.tr_frame,
                                     text="About",
                                     command=self.abw)
        self.train_aboutBtn.grid(row=1,
                                 column=10,
                                 columnspan=1,
                                 sticky="W",
                                 padx=(30, 1),
                                 pady=1)

        self.train_helpBtn = Button(self.tr_frame,
                                     text="Help",
                                     command=self.hlp)
        self.train_helpBtn.grid(row=2,
                                 column=10,
                                 columnspan=1,
                                 sticky="W",
                                 padx=(30, 1),
                                 pady=1)

        self.tnb_helpBtn = Button(self.tr_frame,
                                    text="TB",
                                    command=self.tnb)
        self.tnb_helpBtn.grid(row=3,
                            column=10,
                            columnspan=1,
                            sticky="W",
                            padx=(30, 1),
                            pady=1)

        self.flairTagLbl = Label(self.tr_frame, text="General Input Folder:")
        self.flairTagLbl.grid(row=3, column=0, sticky='E', padx=5, pady=2)
        self.flairTxt = Entry(self.tr_frame,state='disabled',
                              textvariable=self.setting_gen_tag)
        self.flairTxt.grid(row=3, column=1, columnspan=1, sticky="W", pady=1)




        self.modelTagLbl = Label(self.model_frame, text="Model name:")
        self.modelTagLbl.grid(row=6, column=0,
                              sticky='W', padx=(1,1), pady=1)

        self.modelTxt = Entry(self.model_frame,
                              textvariable=self.setting_net_name)
        self.modelTxt.grid(row=6, column=0, columnspan=1, sticky="W", padx=(100,1), pady=1)

        self.GUCM = Checkbutton(self.model_frame,
                                    text="GUCM",
                                    var=self.GUCM_train)
        self.GUCM.grid(row=7, column=0, sticky='W', padx=1,
                            pady=1)

        self.GRCM = Checkbutton(self.model_frame,
                                            text="GRCM",
                                            var=self.GRCM_train)
        self.GRCM.grid(row=7, column=0, sticky='W', padx=(75,1), pady=1)

        self.GWCM = Checkbutton(self.model_frame,
                               text="GWCM",
                               var=self.GWCM_train)
        self.GWCM.grid(row=7, column=0, sticky='W', padx=(150,1), pady=1)


        self.UUCM = Checkbutton(self.model_frame,
                                text="UUCM",
                                var=self.UUCM_train)
        self.UUCM.grid(row=7, column=0, sticky='W', padx=(225,1), pady=1)

        self.URCM = Checkbutton(self.model_frame,
                            text="URCM",
                            var=self.URCM_train)
        self.URCM.grid(row=7, column=0, sticky='W', padx=(300,1), pady=1)

        self.UWCM = Checkbutton(self.model_frame,
                            text="UWCM",
                            var=self.UWCM_train)
        self.UWCM.grid(row=7, column=0, sticky='W', padx=(375,1), pady=1)

        self.RCON = Checkbutton(self.model_frame,
                                text="RCON",
                                var=self.RCON_train)
        self.RCON.grid(row=7, column=0, sticky='W', padx=(450,1), pady=1)


        # START button links
        self.trainingBtn = Button(self.model_frame,
                                  state='disabled',
                                  text="Run only training",
                                  command=self.train_net)
        self.trainingBtn.grid(row=8, column=0, sticky='W', padx=1, pady=1)

        self.traininginferenceBtn = Button(self.model_frame,
                                  state='disabled',
                                  text="Run training and inference",
                                  command=self.train_test_net)
        self.traininginferenceBtn.grid(row=8, column=0, sticky='W', padx=(138,1), pady=1)

        self.testingBtn = Button(self.model_frame,
                                  state='disabled',
                                  text="Run only inference",
                                  command=self.test_net)
        self.testingBtn.grid(row=8, column=0, sticky='W', padx=(335,1), pady=1)

        img1 = ImageTk.PhotoImage(Image.open('images/img.jpg'))
        imglabel = Label(self.train_frame, image=img1)
        imglabel.image = img1
        imglabel.grid(row=8, column=2, padx=(1, 1), pady=1)
        self.process_indicator = StringVar()
        self.process_indicator.set(' ')
        self.label_indicator = Label(master,
                                     textvariable=self.process_indicator)
        self.label_indicator.pack(side="left")
        self.master.protocol("WM_DELETE_WINDOW", self.terminate)

    def settings(self):

        t = Toplevel(self.master)
        t.wm_title("Additional Parameter Settings")

        # data parameters
        t_data = LabelFrame(t, text="Settings ...")
        t_data.grid(row=0, sticky="WE")
        threshold_label = Label(t_data, text="Threshold:      ")
        threshold_label.grid(row=11, sticky="W")
        threshold_entry = Entry(t_data, textvariable=self.setting_threshold)
        threshold_entry.grid(row=11, column=1, sticky="E")
        vovolume_tolerance_label = Label(t_data, text="Error Tolerance:   ")
        vovolume_tolerance_label.grid(row=13, sticky="W")
        vovolume_tolerance_entry = Entry(t_data, textvariable=self.setting_error_tolerance)
        vovolume_tolerance_entry.grid(row=13, column=1, sticky="E")
        t_model = LabelFrame(t, text="Training:")
        t_model.grid(row=14, sticky="EW")

        maxepochs_label = Label(t_model, text="Max epochs:                  ")
        maxepochs_label.grid(row=15, sticky="W")
        maxepochs_entry = Entry(t_model, textvariable=self.setting_max_epochs)
        maxepochs_entry.grid(row=15, column=1, sticky="E")

        batchsize_label = Label(t_model, text="Test batch size:")
        batchsize_label.grid(row=17, sticky="W")
        batchsize_entry = Entry(t_model, textvariable=self.setting_batch_size)
        batchsize_entry.grid(row=17, column=1, sticky="E")


        mode_label = Label(t_model, text="Verbosity:")
        mode_label.grid(row=18, sticky="W")
        mode_entry = Entry(t_model, textvariable=self.setting_net_verbose)
        mode_entry.grid(row=18, column=1, sticky="E")

        gpu_number = Label(t_model, text="GPU number:")
        gpu_number.grid(row=19, sticky="W")
        gpu_entry = Entry(t_model, textvariable=self.setting_gpu_number)
        gpu_entry.grid(row=19, column=1, sticky="W")

    def load_tensorBoard_path(self):

        initialdir = '/tensorboardlogs' + os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_tensorboard_folder.set(fname)
                self.TensorBoard_inFolderTxt.delete(0, END)
                self.TensorBoard_inFolderTxt.insert(0, self.setting_tensorboard_folder.get())
                self.TensorBoardBtn['state'] = 'normal'
            except:
                pass

    def load_traintest_configuration(self):

        traintest_config = configparser.ConfigParser()
        traintest_config.read(os.path.join(self.path, 'config', 'configuration.cfg'))
        self.setting_training_folder.set(traintest_config.get('traintestset',
                                                          'training_folder'))
        self.setting_cross_validation.set(traintest_config.get('traintestset',
                                                          'cross_validation_folder'))
        self.setting_tensorboard_folder.set(traintest_config.get('tensorboard',
                                                          'tensorBoard_folder'))
        self.setting_PORT_tag.set(traintest_config.getint('tensorboard',
                                                     'port'))

        self.setting_test_folder.set(traintest_config.get('traintestset',
                                                      'inference_folder'))


        self.setting_save_tmp.set(traintest_config.get('traintestset', 'save_tmp'))
        self.setting_debug.set(traintest_config.get('traintestset', 'debug'))
        self.setting_use_learnedmodel_model.set(traintest_config.get('traintestset', 'full_train'))
        self.setting_learnedmodel_model.set(traintest_config.get('traintestset', 'learnedmodel'))
        self.setting_learnedmodel_model.set(traintest_config.get('traintestset', 'learnedmodel'))
        self.setting_inference_model.set("      ")
        self.setting_net_folder = os.path.join(self.current_folder, 'models')
        self.setting_net_name.set(traintest_config.get('traintestset', 'name'))
        self.setting_max_epochs.set(traintest_config.getint('traintestset', 'max_epochs'))
        self.setting_patience.set(traintest_config.getint('traintestset', 'patience'))
        self.setting_batch_size.set(traintest_config.getint('traintestset', 'batch_size'))
        self.setting_net_verbose.set(traintest_config.get('traintestset', 'net_verbose'))
        self.setting_gpu_number.set(traintest_config.getint('traintestset', 'gpu_number'))
        self.setting_threshold.set(traintest_config.getfloat('traintestset',
                                                     'threshold'))
        self.setting_error_tolerance.set(traintest_config.getfloat('traintestset',
                                                     'error_tolerance'))
        self.URCM_train.set(traintest_config.get('nets', 'URCM'))
        self.UUCM_train.set(traintest_config.get('nets', 'UUCM'))
        self.UWCM_train.set(traintest_config.get('nets', 'UWCM'))
        self.GRCM_train.set(traintest_config.get('nets', 'GRCM'))
        self.GUCM_train.set(traintest_config.get('nets', 'GUCM'))
        self.GWCM_train.set(traintest_config.get('nets', 'GWCM'))
        self.RCON_train.set(traintest_config.get('nets', 'RCON'))

    def updated_traintest_configuration(self):

        traintest_config = configparser.ConfigParser()
        traintest_config.read(os.path.join(self.path, 'config', 'configuration.cfg'))
    def write_user_configuration(self):

        user_config = configparser.ConfigParser()
        user_config.add_section('traintestset')

        user_config.set('traintestset', 'training_folder', self.setting_training_folder.get())
        user_config.set('traintestset', 'cross_validation_folder', self.setting_cross_validation.get())

        user_config.set('traintestset', 'inference_folder', self.setting_test_folder.get())
        user_config.set('traintestset', 'save_tmp', str(self.setting_save_tmp.get()))
        user_config.set('traintestset', 'debug', str(self.setting_debug.get()))

        user_config.set('traintestset',
                        'full_train',
                        str(not (self.setting_use_learnedmodel_model.get())))
        user_config.set('traintestset',
                        'learnedmodel_model',
                        str(self.setting_learnedmodel_model.get()))

        user_config.set('traintestset', 'name', self.setting_net_name.get())
        user_config.set('traintestset', 'learnedmodel', str(self.setting_learnedmodel))

        user_config.set('traintestset', 'max_epochs', str(self.setting_max_epochs.get()))
        user_config.set('traintestset', 'patience', str(self.setting_patience.get()))
        user_config.set('traintestset', 'batch_size', str(self.setting_batch_size.get()))
        user_config.set('traintestset', 'net_verbose', str(self.setting_net_verbose.get()))
        # user_config.set('model', 'gpu_mode', self.setting_mode.get())
        user_config.set('traintestset', 'gpu_number', str(self.setting_gpu_number.get()))
        user_config.set('traintestset', 'threshold', str(self.setting_threshold.get()))

        user_config.set('traintestset',
                        'error_tolerance', str(self.setting_error_tolerance.get()))
        user_config.add_section('tensorboard')
        user_config.set('tensorboard', 'port', str(self.setting_PORT_tag.get()))
        # postprocessing parameters
        user_config.set('tensorboard', 'tensorBoard_folder', self.setting_tensorboard_folder.get())

        user_config.add_section('nets')
        user_config.set('nets', 'URCM', str(self.URCM_train.get()))
        user_config.set('nets', 'UUCM', str(self.UUCM_train.get()))
        user_config.set('nets', 'UWCM', str(self.UWCM_train.get()))
        user_config.set('nets', 'GRCM', str(self.GRCM_train.get()))
        user_config.set('nets', 'GUCM', str(self.GUCM_train.get()))
        user_config.set('nets', 'GWCM', str(self.GWCM_train.get()))
        user_config.set('nets', 'RCON', str(self.RCON_train.get()))

        with open(os.path.join(self.path,
                               'config',
                               'configuration.cfg'), 'w') as configfile:
            user_config.write(configfile)

    def write_user_configuration_inference(self):

        user_config = configparser.ConfigParser()

        user_config.add_section('traintestset')
        user_config.set('traintestset', 'training_folder', self.setting_training_folder.get())
        user_config.set('traintestset', 'cross_validation_folder', self.setting_cross_validation.get())

        user_config.set('traintestset', 'inference_folder', self.setting_test_folder.get())
        user_config.set('traintestset', 'save_tmp', str(self.setting_save_tmp.get()))
        user_config.set('traintestset', 'debug', str(self.setting_debug.get()))

        user_config.set('traintestset',
                        'full_train',
                        str(not (self.setting_use_learnedmodel_model.get())))
        user_config.set('traintestset',
                        'learnedmodel_model',
                        str(self.setting_learnedmodel_model.get()))


        user_config.set('traintestset', 'name', self.setting_net_name.get())

        user_config.set('traintestset', 'learnedmodel', str(self.setting_learnedmodel))

        user_config.set('traintestset', 'max_epochs', str(self.setting_max_epochs.get()))
        user_config.set('traintestset', 'patience', str(self.setting_patience.get()))
        user_config.set('traintestset', 'batch_size', str(self.setting_batch_size.get()))
        user_config.set('traintestset', 'net_verbose', str(self.setting_net_verbose.get()))
        user_config.set('traintestset', 'gpu_number', str(self.setting_gpu_number.get()))
        user_config.set('traintestset', 'threshold', str(self.setting_threshold.get()))

        user_config.set('traintestset',
                        'error_tolerance', str(self.setting_error_tolerance.get()))

        user_config.add_section('tensorboard')
        user_config.set('tensorboard', 'port', str(self.setting_PORT_tag.get()))
        user_config.set('tensorboard', 'tensorBoard_folder', self.setting_tensorboard_folder.get())
        user_config.add_section('nets')
        user_config.set('nets', 'URCM', str(self.URCM_train.get()))
        user_config.set('nets', 'UUCM', str(self.UUCM_train.get()))
        user_config.set('nets', 'UWCM', str(self.UWCM_train.get()))
        user_config.set('nets', 'GRCM', str(self.GRCM_train.get()))
        user_config.set('nets', 'GUCM', str(self.GUCM_train.get()))
        user_config.set('nets', 'GWCM', str(self.GWCM_train.get()))
        user_config.set('nets', 'RCON', str(self.RCON_train.get()))

        with open(os.path.join(self.path,
                               'config',
                               'configuration.cfg'), 'w') as configfile:
            user_config.write(configfile)

    def load_training_path(self):
        initialdir = '/data' + os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_training_folder.set(fname)
                self.inFolderTxt.delete(0, END)
                self.inFolderTxt.insert(0, self.setting_training_folder.get())

            except:
                pass

    def load_cross_validation_path(self):
        initialdir = '/data' +  os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_cross_validation.set(fname)
                self.incvFolderTxt.delete(0, END)
                self.incvFolderTxt.insert(0, self.setting_cross_validation.get())
                self.trainingBtn['state'] = 'normal'

            except:
                pass

    def load_testing_path(self):
        initialdir = '/data' + os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_test_folder.set(fname)
                self.test_inFolderTxt.delete(0, END)
                self.test_inFolderTxt.insert(0, self.setting_test_folder.get())
                self.testingBtn['state'] = 'normal'
                if self.trainingBtn['state'] == 'normal':
                    self.traininginferenceBtn['state'] = 'normal'
            except:
                pass

    def update_learnedmodel_nets(self):
        folders = os.listdir(self.setting_net_folder)
        self.list_train_learnedmodel_nets = folders
        self.list_test_nets = folders

    def write_to_console(self, txt):
        self.command_out.insert(END, str(txt))

    def write_to_train_test_console(self, txt):
        self.command_out_tt.insert(END, str(txt))

    def write_to_test_console(self, txt):
        self.test_command_out.insert(END, str(txt))

    def start_tensorBoard(self):

            try:
                if self.setting_PORT_tag.get() == None:
                    print("\n")
            except ValueError:
                print("ERROR: Port number and TensorBoard folder must be defined  before starting...\n")
                return

            self.TensorBoardBtn['state'] = 'disable'

            if self.setting_PORT_tag.get() is not None:
                # self.TensorBoardBtn['state'] = 'normal'
                print("\n-----------------------")
                print("Starting TensorBoard ...")
                print("TensorBoard folder:", self.setting_tensorboard_folder.get(), "\n")
                thispath = self.setting_tensorboard_folder.get()
                thisport = self.setting_PORT_tag.get()
                self.write_user_configuration()
                print("The port for TensorBoard is set to be:", thisport)
                # import appscript
                pp = os.path.join(self.path, 'spider', 'bin')

                THIS_PATHx = os.path.split(os.path.realpath(__file__))[0]
                # tensorboard = THIS_PATHx + '/libs/bin/tensorboard'
                Folder=thispath
                Port=thisport
                os_host = platform.system()
                if os_host == 'Windows':
                    arg1 = ' ' + '--logdir  ' + str(Folder) + ' ' + '  --port  ' + str(Port)
                    os.system("start cmd  /c   'tensorboard   {}'".format(arg1))
                elif os_host == 'Linux':
                    arg1 =str(Folder)+'  ' + str(Port)
                    os.system("dbus-launch gnome-terminal -e 'bash -c \"bash  tensorb.sh   {}; exec bash\"'".format(arg1))

                elif os_host == 'Darwin':
                    import appscript
                    appscript.app('Terminal').do_script(
                        'tensorboard    --logdir=' + str(
                            thispath) + '  --port=' + str(thisport))

                else:
                    print("> ERROR: The OS system", os_host, "is not currently supported.")


    def test_net(self):

        if self.setting_net_name.get() == 'None' or self.setting_net_name.get() == '':

            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, define network name before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.URCM_train.get() == False and self.UUCM_train.get() == False and self.UWCM_train.get() == False and  self.GRCM_train.get() == False and self.GUCM_train.get() == False and self.GWCM_train.get() == False and self.RCON_train.get() == False:
            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, check/set (URCM,UUCM ... )  before starting..." + '\x1b[0m')
            print("\n")

            return

        count = 0
        if self.URCM_train.get() == True:
            count = count + 1
        if self.UUCM_train.get() == True:
            count = count + 1
        if self.UWCM_train.get() == True:
            count = count + 1
        if self.GRCM_train.get() == True:
            count = count + 1
        if self.GUCM_train.get() == True:
            count = count + 1
        if self.GWCM_train.get() == True:
            count = count + 1
        if self.RCON_train.get() == True:
            count = count + 1
        if count > 1:

            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, chose between one of the possible models: URCM, UUCM ...   before starting!" + '\x1b[0m')
            print("\n")

            return

        if self.testing_do is None:
            self.testingBtn.config(state='disabled')
            self.traininginferenceBtn['state'] = 'disable'
            self.trainingBtn['state'] = 'disable'
            self.traininginferenceBtn.update()
            self.trainingBtn.update()
            self.updated_traintest_configuration
            self.setting_net_name.set(self.setting_net_name.get())
            self.setting_use_learnedmodel_model.set(False)
            self.write_user_configuration_inference()
            self.testing_do = ThreadedTask(self.write_to_test_console,
                                          self.test_queue, mode='testing')
            self.testing_do.start()

            self.master.after(100, self.process_run)
            self.testingBtn['state'] = 'normal'

    def train_test_net(self):
        if self.setting_net_name.get() == 'None' or self.setting_net_name.get() == '':

            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, define network name before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.URCM_train.get() == False and self.UUCM_train.get() == False and self.UWCM_train.get() == False and  self.GRCM_train.get() == False and self.GUCM_train.get() == False and self.GWCM_train.get() == False and self.RCON_train.get() == False:
            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, check/set (URCM,UUCM ... )  before starting..." + '\x1b[0m')
            print("\n")

            return
        count = 0
        if self.URCM_train.get() == True:
            count = count + 1
        if self.UUCM_train.get() == True:
            count = count + 1
        if self.UWCM_train.get() == True:
            count = count + 1
        if self.GRCM_train.get() == True:
            count = count + 1
        if self.GUCM_train.get() == True:
            count = count + 1
        if self.GWCM_train.get() == True:
            count = count + 1
        if self.RCON_train.get() == True:
            count = count + 1
        if count > 1:

            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, chose between one of the possible models: URCM, UUCM ...   before starting!" + '\x1b[0m')
            print("\n")

            return

        self.traininginferenceBtn['state'] = 'disable'
        self.trainingBtn['state'] = 'disable'
        self.testingBtn['state'] = 'disable'

        if self.training_do is None:
            self.traininginferenceBtn.update()
            self.trainingBtn.update()
            self.testingBtn.update()
            self.write_user_configuration()
            self.training_do = ThreadedTask(self.write_to_train_test_console,
                                           self.test_queue,
                                           mode='trainingandinference')
            self.training_do.start()
            self.master.after(100, self.process_run)

    def train_net(self):

        if self.setting_net_name.get() == 'None' or self.setting_net_name.get() == '':

            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, define network name before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.URCM_train.get() == False and self.UUCM_train.get() == False and self.UWCM_train.get() == False and  self.GRCM_train.get() == False and self.GUCM_train.get() == False and self.GWCM_train.get() == False and self.RCON_train.get() == False:
            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, check/set (URCM,UUCM ... )  before starting..." + '\x1b[0m')
            print("\n")

            return
        count = 0
        if self.URCM_train.get() == True:
            count = count + 1
        if self.UUCM_train.get() == True:
            count = count + 1
        if self.UWCM_train.get() == True:
            count = count + 1
        if self.GRCM_train.get() == True:
            count = count + 1
        if self.GUCM_train.get() == True:
            count = count + 1
        if self.GWCM_train.get() == True:
            count = count + 1
        if self.RCON_train.get() == True:
            count = count + 1
        if count > 1:

            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + " Please, chose between one of the possible models: URCM, UUCM ...   before starting!" + '\x1b[0m')
            print("\n")

            return

        self.traininginferenceBtn['state'] = 'disable'
        self.trainingBtn['state'] = 'disable'
        self.testingBtn['state'] = 'disable'

        if self.training_do is None:
            self.traininginferenceBtn.update()
            self.trainingBtn.update()
            self.testingBtn.update()
            self.write_user_configuration()
            self.training_do = ThreadedTask(self.write_to_console,
                                           self.test_queue,
                                           mode='training')
            self.training_do.start()
            self.master.after(100, self.process_run)

    def abw(self):

        t = Toplevel(self.master, width=500, height=500)
        t.wm_title("Multi task 3D Convolutional Neural Network")
        title = Label(t,
                      text="Variational Auto Encoder (VAE)\n"
                      "Gaussian Uniform Boltzmann \n"
                      "Engineering - UCL \n"
                      "Kevin Bronik - 2021")
        title.grid(row=2, column=1, padx=20, pady=10)
        img = ImageTk.PhotoImage(Image.open('images/1.jpg'))
        imglabel = Label(t, image=img)
        imglabel.image = img
        imglabel.grid(row=1, column=1, padx=10, pady=10)
        root = imglabel
        root.mainloop()
        # self.gif = tk.PhotoImage(file=self.gif_file,
        # root = imglabel

        # Add the path to a GIF to make the example working
        # l = AnimatedGIF(root, "images/brain_lesion.gif")
        # # l = AnimatedGIF(root, "./brain_lesion.png")
        # l.pack()
        # root.mainloop()

    def hlp(self):
            t = Toplevel(self.master, width=500, height=500)

            img = ImageTk.PhotoImage(Image.open('images/help.jpg'))
            imglabel = Label(t, image=img)
            imglabel.image = img
            imglabel.grid(row=1, column=1, padx=10, pady=10)
            # self.gif = tk.PhotoImage(file=self.gif_file,
            root = imglabel
            root.mainloop()

    def tnb(self):
        t = Toplevel(self.master, width=500, height=500)
        t.wm_title("TensorBoard Settings")
        self.TensorBoard_inFolderLbl = Label(t, text="TensorBoard folder:")
        self.TensorBoard_inFolderLbl.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        self.TensorBoard_inFolderTxt = Entry(t)
        self.TensorBoard_inFolderTxt.grid(row=1,
                                          column=1,
                                          columnspan=5,
                                          sticky="W",
                                          pady=3)
        self.TensorBoard_inFileBtn = Button(t, text="Browse ...",
                                            command=self.load_tensorBoard_path)
        self.TensorBoard_inFileBtn.grid(row=2,
                                        column=1,
                                        columnspan=1,
                                        sticky='W',
                                        padx=5,
                                        pady=1)
        self.portTagLbl = Label(t, text="Port:")
        self.portTagLbl.grid(row=3, column=0, sticky='E', padx=5, pady=2)
        self.portTxt = Entry(t,
                             textvariable=self.setting_PORT_tag)
        self.portTxt.grid(row=3, column=1, columnspan=1, sticky="W", pady=1)


        self.TensorBoardBtn = Button(t,
                                     state='disabled',
                                     text="Start TensorBoard",
                                     command=self.start_tensorBoard)
        self.TensorBoardBtn.grid(row=4, column=1, sticky='W', padx=1, pady=1)


    def process_run(self):

        self.process_indicator.set('Training/Testing is Running... please wait')
        try:
            msg = self.test_queue.get(0)
            self.process_indicator.set('Training/Testing completed.')

            self.trainingBtn['state'] = 'normal'
            self.traininginferenceBtn['state'] = 'normal'
            # self.testingBtn['state'] = 'normal'
        except queue.Empty:
            self.master.after(100, self.process_run)

    def terminate(self):

        if self.training_do is not None:
            self.training_do.stop_process()
        if self.testing_do is not None:
            self.testing_do.stop_process()
        os.system('cls' if platform.system == "Windows" else 'clear')
        root.destroy()


class ThreadedTask(threading.Thread):

    def __init__(self, print_func, queue, mode):
        threading.Thread.__init__(self)
        self.queue = queue
        self.mode = mode
        self.print_func = print_func
        self.process = None

    def run(self):

        settings = overall_config()
        if self.mode == 'training':
                print('\x1b[6;30;41m' + "                                       " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting  VAE  training ...            " + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                       " + '\x1b[0m')
                train_network_vae(settings)

        elif self.mode == 'trainingandinference':
                print('\x1b[6;30;41m' + "                                                   " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting VAE training and inference ...            " + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                                   " + '\x1b[0m')
                train_test_network_vae(settings)
        else:
                print('\x1b[6;30;41m' + "                                       " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting  VAE  inference ...           " + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                       " + '\x1b[0m')
                infer_vae(settings)
        self.queue.put(" ")

    def stop_process(self):
        try:
            if platform.system() == "Windows":
                subprocess.Popen("taskkill /F /T /PID %i" % os.getpid(), shell=True)
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        except:
            os.kill(os.getpid(), signal.SIGTERM)

if __name__ == '__main__':

    try:
        print('')
        print('')
        print('\x1b[6;30;42m' + 'Training started.......................' + '\x1b[0m')
        parser = argparse.ArgumentParser()
        parser.add_argument('--docker',
                            dest='docker',
                            action='store_true')
        parser.set_defaults(docker=False)
        args = parser.parse_args()
        root = Tk()
        root.resizable(width=False, height=False)
        GUI = CNN_VAE(root, args.docker)
        root.mainloop()
        print('\x1b[6;30;42m' + 'Training completed.....................' + '\x1b[0m')
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)