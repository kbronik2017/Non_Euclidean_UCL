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
#     {Variational Auto Encoder (VAE) Gaussian Boltzmann}
#     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.


from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, Activation, Lambda,Flatten,Dense,Reshape
from keras.layers.merge import Concatenate
from keras.applications import VGG16
from keras import backend as K
from libs.pconv_layer import PConv2D
from tensorflow.python.ops import math_ops

# Weighted configuration model (WCM)

def get_network_Gaussian_Boltzmann_WCM(img_rows, img_cols, latent_dim):
#     img_rows = 400
#     img_cols = 400

    st = 0
    inputs_img = Input((img_rows, img_cols, 3), name='inputs_img')
    inputs_mask = Input((img_rows, img_cols, 3), name='inputs_mask')

# 1

    convolution1, mask1 = PConv2D(64, 7, strides=2, padding='same')([inputs_img, inputs_mask])
    convolution1 = Activation('relu')(convolution1)
    st += 1

# 2

    convolution2, mask2 = PConv2D(128, 5, strides=2, padding='same')([convolution1, mask1])
    convolution2 = BatchNormalization(name='BatchNormalization'+str(st))(convolution2, training=True)
    convolution2 = Activation('relu')(convolution2)
    st += 1

# 3
    convolution3, mask3 = PConv2D(256, 5, strides=2, padding='same')([convolution2, mask2])
    convolution3 = BatchNormalization(name='BatchNormalization'+str(st))(convolution3, training=True)
    convolution3 = Activation('relu')(convolution3)
    st += 1

# 4
    convolution4, mask4 = PConv2D(512, 3, strides=2, padding='same')([convolution3, mask3])
    convolution4 = BatchNormalization(name='BatchNormalization'+str(st))(convolution4, training=True)
    convolution4 = Activation('relu')(convolution4)
    st += 1

# 5
    convolution5, mask5 = PConv2D(512, 3, strides=5, padding='same')([convolution4, mask4])
    convolution5 = BatchNormalization(name='BatchNormalization'+str(st))(convolution5, training=True)
    convolution5 = Activation('relu')(convolution5)
    st += 1

# 6
    convolution6, mask6 = PConv2D(512, 3, strides=1, padding='same')([convolution5, mask5])
    convolution6 = BatchNormalization(name='BatchNormalization'+str(st))(convolution6, training=True)
    convolution6 = Activation('relu')(convolution6)
    st += 1

# 6
    con_dense1 = Flatten()(convolution6)
# 7
    con_dense2 = Dense(6400, activation='relu')(con_dense1)

# 8
    con_dense3 = Dense(3200, activation='relu')(con_dense2)


# 9
    z_mean = Dense(latent_dim, name='z_mean')(con_dense3)
    z_log_var = Dense(latent_dim, name='z_log_var')(con_dense3)

    def sampling(args):
       z_mean, z_log_var = args
       epsilon = K.random_normal(shape=K.shape(z_mean))
    #            epsilon = K.random_normal(shape=(2508,), mean=0.,
    #                                      stddev=1.0)
       return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(inputs=[inputs_img, inputs_mask], outputs=[z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    decoder_input = Input(shape=(latent_dim,))
    dcon_dense3 = Dense(3200)(decoder_input)
    dcon_dense3 = LeakyReLU(alpha=0.2)(dcon_dense3)
# 2

    dcon_dense2 = Dense(6400)(dcon_dense3)
    dcon_dense2 = LeakyReLU(alpha=0.2)(dcon_dense2)

# 3
    dcon_dense1 = Dense(12800)(dcon_dense2)
    dcon_dense1 = LeakyReLU(alpha=0.2)(dcon_dense1)
    dcon_dense1 = Reshape((5,5,512))(dcon_dense1)

# 4
    up_img = UpSampling2D(size=(1,1))(dcon_dense1)
    up_mask = UpSampling2D(size=(1,1))(mask6)
    concat_img1 = Concatenate(axis=3)([convolution5,up_img])
    concat_mask1 = Concatenate(axis=3)([mask5,up_mask])
    deconvolution1, dmask1 = PConv2D(512, 3, padding='same')([concat_img1, concat_mask1])
    deconvolution1 = BatchNormalization()(deconvolution1)
    deconvolution1 = LeakyReLU(alpha=0.2)(deconvolution1)

# 5
    up_img = UpSampling2D(size=(5,5))(deconvolution1)
    up_mask = UpSampling2D(size=(5,5))(dmask1)
    concat_img2 = Concatenate(axis=3)([convolution4,up_img])
    concat_mask2 = Concatenate(axis=3)([mask4,up_mask])
    deconvolution2, dmask2 = PConv2D(512, 3, padding='same')([concat_img2, concat_mask2])
    deconvolution2 = BatchNormalization()(deconvolution2)
    deconvolution2 = LeakyReLU(alpha=0.2)(deconvolution2)

# 6
    up_img = UpSampling2D(size=(2,2))(deconvolution2)
    up_mask = UpSampling2D(size=(2,2))(dmask2)
    concat_img3 = Concatenate(axis=3)([convolution3,up_img])
    concat_mask3 = Concatenate(axis=3)([mask3,up_mask])
    deconvolution3, dmask3 = PConv2D(256, 5, padding='same')([concat_img3, concat_mask3])
    deconvolution3 = BatchNormalization()(deconvolution3)
    deconvolution3 = LeakyReLU(alpha=0.2)(deconvolution3)

# 7
    up_img = UpSampling2D(size=(2,2))(deconvolution3)
    up_mask = UpSampling2D(size=(2,2))(dmask3)
    concat_img4 = Concatenate(axis=3)([convolution2,up_img])
    concat_mask4 = Concatenate(axis=3)([mask2,up_mask])
    deconvolution4, dmask4 = PConv2D(128, 3, padding='same')([concat_img4, concat_mask4])
    deconvolution4 = BatchNormalization()(deconvolution4)
    deconvolution4 = LeakyReLU(alpha=0.2)(deconvolution4)

# 8
    up_img = UpSampling2D(size=(2,2))(deconvolution4)
    up_mask = UpSampling2D(size=(2,2))(dmask4)
    concat_img5 = Concatenate(axis=3)([convolution1,up_img])
    concat_mask5 = Concatenate(axis=3)([mask1,up_mask])
    deconvolution5, dmask5 = PConv2D(64, 3, padding='same')([concat_img5, concat_mask5])
    deconvolution5 = BatchNormalization()(deconvolution5)
    deconvolution5 = LeakyReLU(alpha=0.2)(deconvolution5)

# 9
    up_img = UpSampling2D(size=(2,2))(deconvolution5)
    up_mask = UpSampling2D(size=(2,2))(dmask5)
    concat_img6 = Concatenate(axis=3)([inputs_img,up_img])
    concat_mask6 = Concatenate(axis=3)([inputs_mask,up_mask])
    deconvolution6, dmask6 = PConv2D(3, 3, padding='same')([concat_img6, concat_mask6])
    deconvolution6 = LeakyReLU(alpha=0.2)(deconvolution6)

# 10

    decoder_outputs1 = Conv2D(3, 1, activation='exponential', name='outputs_img1')(deconvolution6)
    decoder_outputs2 = Conv2D(3, 1, activation='exponential', name='outputs_img2')(deconvolution6)



    decoder = Model(inputs=[inputs_img, inputs_mask, decoder_input], outputs=[decoder_outputs1, decoder_outputs2] , name="decoder")
    decoder.summary()


    encoded_img1, encoded_img2, encoded_img3 = encoder([inputs_img, inputs_mask])
    decoded_img1, decoded_img2 = decoder([inputs_img, inputs_mask,encoded_img3])
    autoencoder = Model([inputs_img, inputs_mask], [decoded_img1, decoded_img2], name="autoencoder")
    autoencoder.summary()


    return autoencoder, encoder, decoder
