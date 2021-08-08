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
#     {Variational Auto Encoder (VAE) Uniform Boltzmann}
#     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.


import numpy as np

from keras import backend as K

from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, \
    Conv2DTranspose, MaxPooling2D, UpSampling2D, Lambda, LeakyReLU

from keras.models import Model

import tensorflow as tf


# Reciprocal configuration model (RCM)

def get_network_Uniform_Boltzmann_RCM(img_width, img_height, latentDim):

    input_img = Input(shape=(img_width, img_height, 3))
    x = input_img
    # Encoding network
    # x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # encoder_OUT = MaxPooling2D((2, 2), padding='same')(x)
    filters = (16, 32, 64)
    CDim = -1

    for f in filters:
        # apply a CONV => RELU => BN operation
        x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=CDim)(x)
        # x = Dropout(0.2)(x)
    VSize = K.int_shape(x)

    x = Flatten()(x)
    # latent = Dense(latentDim)(x)
    a = Dense(units=latentDim, name='z_mean')(x)
    # z_mean = Dense(units=latentDim, name='z_mean', kernel_regularizer=regularizers.l2(0.0002))(x)
    b = Dense(units=latentDim, name='z_log_var')(x)
    @tf.function
    def sampling(args):
        args1, arg2 = args
        # epsilon = K.random_normal(shape=K.shape(args1))
        epsilon = K.random_uniform(shape=K.shape(args1))
        if tf.reduce_mean(tf.reduce_sum(arg2)) - tf.reduce_mean(tf.reduce_sum(args1)) != 0:
            return args1 + (arg2 - args1) * epsilon
        else:
            return arg2 * epsilon

        # return args1
        # return epsilon

    # def sampling(args):
    #     (z_mean, z_var) = args
    #     epsilon = K.random_normal(shape=(K.shape(z_mean)[0],
    #                                      latentDim), mean=0., stddev=1.)
    #     return z_mean + z_var * epsilon
    #
    #
    # z = sampling(z_mean, z_log_var)
    z = Lambda(sampling, output_shape=(latentDim,))([a, b])
    # build the encoder model
    encoder = Model(input_img, [a, b, z], name="encoder")
    encoder.summary()

    latentInputs = Input(shape=(latentDim,))
    # x = Dense(np.prod(volumeSize[1:]), kernel_regularizer=regularizers.l2(0.0002))(latentInputs)
    x = Dense(np.prod(VSize[1:]))(latentInputs)
    x = Reshape((VSize[1], VSize[2], VSize[3]))(x)

    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(f, (3, 3), strides=2,
                            padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(axis=CDim)(x)
        # x = Dropout(0.5)(x)
    # decoded1 = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding="same")(x)
    # decoded2 = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding="same")(x)

    decoded1 = Conv2DTranspose(3, (3, 3), activation='exponential', padding="same")(x)
    decoded2 = Conv2DTranspose(3, (3, 3), activation='exponential', padding="same")(x)
    decoded3 = Conv2DTranspose(3, (3, 3), activation='exponential', padding="same")(x)
    decoded4 = Conv2DTranspose(3, (3, 3), activation='exponential', padding="same")(x)
    decoded5 = Conv2DTranspose(3, (3, 3), activation='exponential', padding="same")(x)
    decoded6 = Conv2DTranspose(3, (3, 3), activation='exponential', padding="same")(x)
    # decoded = Conv2DTranspose(3, (3, 3), activation='linear', padding="same")(x)
    # decoded = Conv2DTranspose(3, (3, 3), activation='softmax', padding="same")(x)


    # decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    decoder = Model(latentInputs, [decoded1, decoded2, decoded3, decoded4, decoded5, decoded6], name="decoder")
    # decoder = Model(inputs=decoder_input, outputs=decoded, name="decoder")
    decoder.summary()
    encoded_img1, encoded_img2, encoded_img3 = encoder(input_img)
    # encoded_img1, encoded_img2 = encoder(input_img)
    decoded_img1, decoded_img2, decoded_img3, decoded_img4, decoded_img5, decoded_img6 = decoder(encoded_img3)
    autoencoder_cnn = Model(inputs=input_img, outputs=[decoded_img1, decoded_img2, decoded_img3, decoded_img4,
                                                       decoded_img5, decoded_img6], name="autoencoder")


    return autoencoder_cnn, encoder, decoder
