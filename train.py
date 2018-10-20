from model import *
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, LearningRateScheduler
import numpy as np
import h5py
import random
import os
import keras.backend as K
import math
import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

BATCH_SIZE = 16
EPOCHS = 300
PATCH_SIZE = 48
UPSCALE_FACTOR = 2
NUM_CHANNELS = 3

MEAN_IMAGE_INPUT = cv2.imread('mean_input_image.png', 1)
MEAN_IMAGE_OUTPUT = cv2.imread('mean_output_image.png', 1)

TRAINING_H5_PATH = 'training_h5'
VALIDATION_H5_PATH = 'validation_h5'

TOTAL_DATA_NUMBER = h5py.File(TRAINING_H5_PATH + '/inputs.h5', 'r')['inputs'].shape[0]
training_file_inputs = h5py.File(TRAINING_H5_PATH + '/inputs.h5', 'r')
training_file_outputs = h5py.File(TRAINING_H5_PATH + '/outputs.h5', 'r')
training_data_inputs = training_file_inputs['inputs']
training_data_outputs = training_file_outputs['outputs']


def create_validation_data():
    with h5py.File(VALIDATION_H5_PATH + '/inputs.h5', 'a') as hf:
        input_validation = np.array(hf["inputs"])
    with h5py.File(VALIDATION_H5_PATH + '/outputs.h5', 'a') as hf:
        output_validation = np.array(hf["outputs"])

    input_validation = input_validation.astype('float32')
    output_validation = output_validation.astype('float32')
    input_validation /= 255.
    output_validation /= 255.
    input_validation = input_validation - MEAN_IMAGE_INPUT.astype('float32') / 255.
    output_validation = output_validation - MEAN_IMAGE_OUTPUT.astype('float32') / 255.

    input_validation = input_validation.reshape([-1, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS])
    output_validation = output_validation.reshape(
        [-1, UPSCALE_FACTOR * PATCH_SIZE, UPSCALE_FACTOR * PATCH_SIZE, NUM_CHANNELS])

    return input_validation, output_validation


validation_inputs, validation_outputs = create_validation_data()


def dataGenerator():
    while 1:
        list_index_training = range(0, TOTAL_DATA_NUMBER - 1)
        for i in range(TOTAL_DATA_NUMBER / BATCH_SIZE):
            random_indices = list(np.sort([list_index_training.pop(random.randrange(len(list_index_training))) for _ in
                                           xrange(BATCH_SIZE)]))

            X_train = np.array(training_data_inputs[random_indices])
            y_train = np.array(training_data_outputs[random_indices])
            X_train = X_train.astype('float32')
            y_train = y_train.astype('float32')
            y_train /= 255.
            X_train /= 255.
            X_train = X_train - MEAN_IMAGE_INPUT.astype('float32') / 255.
            y_train = y_train - MEAN_IMAGE_OUTPUT.astype('float32') / 255.

            X_train = X_train.reshape([-1, PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS])
            y_train = y_train.reshape([-1, UPSCALE_FACTOR * PATCH_SIZE, UPSCALE_FACTOR * PATCH_SIZE, NUM_CHANNELS])

            yield X_train, y_train


MEAN_PSNR_EPOCHS = []
MEAN_SSIM_EPOCHS = []
class SaveFilters_And_Get_SSIM_PSNR_Callback(Callback):
    def on_train_begin(self, logs={}):
        save_filters(self.model)
        return

    def on_epoch_end(self, epoch, logs={}):
        save_filters(self.model, Before_Training=False)
        mean_psnr, mean_ssim = get_psnr_ssim_on_validation(cnn_model, validation_inputs, validation_outputs)
        MEAN_PSNR_EPOCHS.append(mean_psnr)
        MEAN_SSIM_EPOCHS.append(mean_ssim)
        plt.close()
        plt.plot(MEAN_PSNR_EPOCHS, color='r')
        plt.xlabel('EPOCHS')
        plt.ylabel('PSNR (db)')
        plt.title('PSNR of Validation Data')
        plt.savefig('validation_psnr.png')
        plt.close()
        plt.plot(MEAN_SSIM_EPOCHS, color='g')
        plt.xlabel('EPOCHS')
        plt.ylabel('SSIM')
        plt.title('SSIM of Validation Data')
        plt.savefig('validation_ssim.png')
        plt.close()
        print 'Mean PSNR on Validation Dataset: ' + str(mean_psnr)
        print 'Mean SSIM on Validation Dataset: ' + str(mean_ssim)
        print '-----------------------------------------------'
        return


def get_psnr_ssim_on_validation(cnn_model, validation_inputs, validation_outputs):
    print 'CALCULATING PSNR AND SSIM ON VALIDATION DATASET'
    prediction = cnn_model.predict(validation_inputs) 
    prediction = np.array(prediction)
    all_psnr = []
    all_ssim = []
    for i in range(validation_outputs.shape[0]):
        i_ssim =  compare_ssim(validation_outputs[i], prediction[i], multichannel=True)
        i_psnr = compare_psnr(validation_outputs[i], prediction[i])

        all_ssim.extend([i_ssim])
        all_psnr.extend([i_psnr])

    mean_psnr = np.mean(all_psnr)
    mean_ssim = np.mean(all_ssim)

    return mean_psnr, mean_ssim


def step_decay(epoch):
    initial_lrate = 1e-4
    drop = 0.5
    epochs_drop = 200.0
    lrate = initial_lrate * drop**math.floor(epoch / epochs_drop)

    return lrate


def PSNR(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true))
    return -10. * (K.log(mse)/K.log(10.))


def set_callbacks():
    if os.path.isdir('Graph'):
        os.system("rm -r Graph")
    if not os.path.isdir('Snapshots'):
        os.mkdir('Snapshots')
    tbCallBack = TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint("./Snapshots/model.h5", verbose=1, monitor='val_loss', save_best_only=False,
                                 mode='auto')
    lrate = LearningRateScheduler(step_decay)
    save_filter_callback = SaveFilters_And_Get_SSIM_PSNR_Callback()

    return checkpoint, lrate, tbCallBack, save_filter_callback



if __name__ == "__main__":
    cnn_model = get_model(PATCH_SIZE, NUM_CHANNELS, UPSCALE_FACTOR)
    adam_optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    cnn_model.compile(loss='mae', optimizer=adam_optimizer, metrics=[PSNR])
    checkpoint, lrate, tbCallBack, save_filter_callback = set_callbacks()
    cnn_model.fit_generator(dataGenerator(), steps_per_epoch=TOTAL_DATA_NUMBER / BATCH_SIZE, nb_epoch=EPOCHS, verbose=1,
                            class_weight=None,
nb_worker=1, callbacks=[checkpoint, lrate, tbCallBack, save_filter_callback])
