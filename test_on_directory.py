from keras.models import load_model
import cv2
import numpy as np
import time
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from train import Subpixel, PSNR
from test_on_single import convert_lr_image_to_hr_image
import glob
import re
import os

PATCH_SIZE = 48
UPSCALE_FACTOR = 2
NUM_CHANNELS = 3
MODEL_PATH = './Snapshots/model.h5'

IMAGES_PATH = 'test_images'
REFERENCE_IMAGE_PATH = 'test_images_ref'
SAVE_DIR = 'results'
MEAN_IMAGE_INPUT_PATH = '.'
MEAN_IMAGE_OUTPUT_PATH = '.'


def find_index_in_HR(image_name, image_directories_HR):
    image_name_reduced = re.sub(r'x[2-9]', '', image_name)
    for index, image_path in enumerate(image_directories_HR):
        if re.search(r'.*{}.*'.format(image_name_reduced), image_path) is not None:
            return index


if __name__ == "__main__":
    model = load_model(MODEL_PATH, custom_objects={"PSNR": PSNR, "Subpixel": Subpixel})
    mean_input_image = cv2.imread(MEAN_IMAGE_INPUT_PATH + '/mean_input_image.png')
    mean_output_image = cv2.imread(MEAN_IMAGE_OUTPUT_PATH + '/mean_output_image.png')
    start = time.time()
    images_psnr = []
    images_ssim = []
    if os.path.isdir(SAVE_DIR) is False:
        os.makedirs(SAVE_DIR)

    for i, lr_image_path in enumerate(glob.glob(IMAGES_PATH + '/*.*')):
        lr_image = cv2.imread(lr_image_path, 1)
        image_name = re.findall(r'.*/(.*)\..*$', lr_image_path)[0]
        image_format = re.findall(r'.*/.*\.(.*)$', lr_image_path)[0]
        hr_image_path = REFERENCE_IMAGE_PATH + '/' + re.sub(r'x2', '', image_name) + '.' + image_format

        network_hr_image = convert_lr_image_to_hr_image(model, lr_image, PATCH_SIZE, NUM_CHANNELS, UPSCALE_FACTOR,
                                                        mean_input_image,
                                                        mean_output_image)
        Reference_image = cv2.imread(hr_image_path, 1)

        Reference_image_y_channel = cv2.cvtColor(Reference_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        network_hr_image_y_channel = cv2.cvtColor(network_hr_image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        # crop Upscale factor pixel from images

        Reference_image_y_channel = Reference_image_y_channel[UPSCALE_FACTOR:Reference_image.shape[0]-UPSCALE_FACTOR, UPSCALE_FACTOR:Reference_image.shape[1]-UPSCALE_FACTOR]
        network_hr_image_y_channel = network_hr_image_y_channel[UPSCALE_FACTOR:network_hr_image_y_channel.shape[0]-UPSCALE_FACTOR, UPSCALE_FACTOR:network_hr_image_y_channel.shape[1]-UPSCALE_FACTOR]


        image_ssim = ssim(Reference_image_y_channel, network_hr_image_y_channel)
        image_psnr = psnr(Reference_image_y_channel, network_hr_image_y_channel)

        images_psnr.append(image_psnr)
        images_ssim.append(image_ssim)

        cv2.imwrite(SAVE_DIR+'/{}.png'.format(i), network_hr_image)

    print 'Processing Time:...'
    processing_time = round(time.time() - start, 2)
    print processing_time
    print 'PSNR: ' + str(np.mean(np.array(images_psnr)))
    print 'SSIM: ' + str(np.mean(np.array(image_ssim)))
