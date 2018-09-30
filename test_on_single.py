from keras.models import load_model
import cv2
import numpy as np
import time
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from train import Subpixel, PSNR

PATCH_SIZE = 48
UPSCALE_FACTOR = 2
NUM_CHANNELS = 3
MODEL_PATH = './Snapshots/model.h5'

IMAGE_PATH = 'Dataset/LR_Validation/DIV2K_valid_LR_bicubic/X2/0801x2.png'
REFERENCE_IMAGE_PATH = 'Dataset/HR_Validation/DIV2K_valid_HR/0801.png'
MEAN_IMAGE_INPUT_PATH = '.'
MEAN_IMAGE_OUTPUT_PATH = '.'


def get_prediction_on_patch(model, patch, patch_size, num_channels, upscale_factor, mean_input_image,
                            mean_output_image):
    patch = patch.astype('float32')
    patch = patch.reshape([1, patch_size, patch_size, num_channels])
    normalized_patch = (patch - mean_input_image) / 255.
    prediction_network = model.predict(normalized_patch)
    prediction_network = prediction_network[0]
    prediction_network = prediction_network.reshape(
        [upscale_factor * PATCH_SIZE, upscale_factor * PATCH_SIZE, num_channels])
    prediction_network = 255. * prediction_network + mean_output_image
    prediction_network[prediction_network > 255] = 255
    prediction_network[prediction_network < 0] = 0
    prediction_network = prediction_network.astype('uint8')

    return prediction_network


def convert_lr_image_to_hr_image(model, lr_image, patch_size, num_channels, upscale_factor, mean_input_image,
                                 mean_output_image):
    output = np.zeros(shape=(UPSCALE_FACTOR * lr_image.shape[0], UPSCALE_FACTOR * lr_image.shape[1], NUM_CHANNELS),
                      dtype='uint8')
    for y in range(0, lr_image.shape[0] - PATCH_SIZE, PATCH_SIZE):
        for x in range(0, lr_image.shape[1] - PATCH_SIZE, PATCH_SIZE):
            patch = np.copy(lr_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :])
            prediction_on_patch = get_prediction_on_patch(model, patch, patch_size, num_channels, upscale_factor,
                                                          mean_input_image, mean_output_image)
            output[upscale_factor * y:upscale_factor * y + upscale_factor * patch_size,
            upscale_factor * x:upscale_factor * x + patch_size * upscale_factor,
            :] = prediction_on_patch
            # cv2.rectangle(lr_image, (x, y), (x+PATCH_SIZE, y+PATCH_SIZE), (0, 255, 0), 3)
            # cv2.rectangle(output, (x*UPSCALE_FACTOR, y*UPSCALE_FACTOR), (x*UPSCALE_FACTOR+PATCH_SIZE*UPSCALE_FACTOR, y*UPSCALE_FACTOR+PATCH_SIZE*UPSCALE_FACTOR), (0, 0, 255), 3)
    # MAKE PREDICTION ON MARGINS
    y = lr_image.shape[0] - PATCH_SIZE
    for x in range(0, lr_image.shape[1] - PATCH_SIZE, PATCH_SIZE):
        patch = np.copy(lr_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :])
        prediction_on_patch = get_prediction_on_patch(model, patch, patch_size, num_channels, upscale_factor,
                                                      mean_input_image, mean_output_image)
        output[upscale_factor * y:upscale_factor * y + upscale_factor * patch_size,
        upscale_factor * x:upscale_factor * x + patch_size * upscale_factor,
        :] = prediction_on_patch

    x = lr_image.shape[1] - PATCH_SIZE
    for y in range(0, lr_image.shape[0] - PATCH_SIZE, PATCH_SIZE):
        patch = np.copy(lr_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :])
        prediction_on_patch = get_prediction_on_patch(model, patch, patch_size, num_channels, upscale_factor,
                                                      mean_input_image, mean_output_image)
        output[upscale_factor * y:upscale_factor * y + upscale_factor * patch_size,
        upscale_factor * x:upscale_factor * x + patch_size * upscale_factor,
        :] = prediction_on_patch


    x = lr_image.shape[1] - PATCH_SIZE
    y = lr_image.shape[0] - PATCH_SIZE
    patch = np.copy(lr_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :])
    prediction_on_patch = get_prediction_on_patch(model, patch, patch_size, num_channels, upscale_factor,
                                                  mean_input_image, mean_output_image)
    output[upscale_factor * y:upscale_factor * y + upscale_factor * patch_size,
    upscale_factor * x:upscale_factor * x + patch_size * upscale_factor,
    :] = prediction_on_patch

    return output


if __name__ == "__main__":
    model = load_model(MODEL_PATH, custom_objects={"PSNR": PSNR, "Subpixel": Subpixel})
    start = time.time()
    lr_image = cv2.imread(IMAGE_PATH, 1)
    mean_input_image = cv2.imread(MEAN_IMAGE_INPUT_PATH + '/mean_input_image.png')
    mean_output_image = cv2.imread(MEAN_IMAGE_OUTPUT_PATH + '/mean_output_image.png')
    network_hr_image = convert_lr_image_to_hr_image(model, lr_image, PATCH_SIZE, NUM_CHANNELS, UPSCALE_FACTOR,
                                                    mean_input_image,
                                                    mean_output_image)
    bicubic_image = cv2.resize(lr_image, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    Reference_image = cv2.imread(REFERENCE_IMAGE_PATH, 1)

    Reference_image_y_channel = cv2.cvtColor(Reference_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    network_hr_image_y_channel = cv2.cvtColor(network_hr_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    # crop Upscale factor pixel from images
    Reference_image_y_channel = Reference_image_y_channel[UPSCALE_FACTOR:Reference_image.shape[0]-UPSCALE_FACTOR, UPSCALE_FACTOR:Reference_image.shape[1]-UPSCALE_FACTOR]
    network_hr_image_y_channel = network_hr_image_y_channel[UPSCALE_FACTOR:network_hr_image_y_channel.shape[0]-UPSCALE_FACTOR, UPSCALE_FACTOR:network_hr_image_y_channel.shape[1]-UPSCALE_FACTOR]

    image_ssim = ssim(Reference_image_y_channel, network_hr_image_y_channel)
    image_psnr = psnr(Reference_image_y_channel, network_hr_image_y_channel)

    print 'Processing Time:...'
    processing_time = round(time.time() - start, 2)
    print processing_time
    print 'PSNR: ' + str(image_psnr)
    print 'SSIM: ' + str(image_ssim)
    cv2.imwrite('LR_Image_{}.png'.format(processing_time), lr_image)
    cv2.imwrite('Bicubic_Image_{}.png'.format(processing_time), bicubic_image)
    cv2.imwrite('Network_Image_{}.png'.format(processing_time), network_hr_image)

    cv2.imshow('LR Image', lr_image)
    cv2.imshow('Bicubic Interpolation', bicubic_image)
    cv2.imshow('Network Prediction', network_hr_image)
    cv2.waitKey(0)
