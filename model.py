from keras.models import Model
from keras.layers import *
from keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from math import sqrt
from SubPixel import Subpixel


def save_filters(cnn_model, Before_Training=True):
    weights = cnn_model.get_weights()[0]  # list of numpy arrays
    weights = np.array(weights)
    for i in range(0, weights.shape[3]):
        kernel = weights[:, :, 0, i]
        plt.subplot(int(np.ceil(sqrt(weights.shape[3]))), int(np.ceil(sqrt(weights.shape[3]))), i + 1)
        plt.imshow(kernel, cmap='gray', interpolation=None)
        plt.axis('off')

    if Before_Training:
        savefig('bt.png')
    else:
        savefig('at.png')



def res_block(inputs):
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    x = Lambda(lambda x: x * 0.1)(x)
    return add([x, inputs])


def up_block(x, upscale_factor):
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    #x = UpSampling2D(size=(upscale_factor, upscale_factor))(x)
    x = Subpixel(3, (3, 3),2, activation=None, padding='same')(x)
    return x


# def subpixel_layer(x):
#    return SubpixelConv2d(x, scale=2, n_out_channel=3)

def get_model(patch_size, num_channels, upscale_factor):
    _input = Input(shape=(patch_size, patch_size, num_channels), name='input')
    x_input_res_block = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(_input)
    x = x_input_res_block
    # add residual blocks
    for i in range(32):
        x = res_block(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(x)
    # skip connection
    x = add([x, x_input_res_block])
    # upscale block
    x = up_block(x, upscale_factor)
    # x = Lambda(subpixel_layer)(x)
    # final conv layer 
    final_out = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), activation=None, padding='same')(x)

    model = Model(input=_input, output=final_out)
    print model.summary()
    plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)

    return model
