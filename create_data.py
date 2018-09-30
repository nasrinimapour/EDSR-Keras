import cv2
import glob
from Database import Database
import re
from tqdm import tqdm
import numpy as np
import random
import imutils

PATCH_SIZE = 48
NUM_CHANNELS = 3
UPSCALE_FACTOR = 2
RELATIVE_TRAINING_IMAGES_PATH_LR = './Dataset/DIV2K_train_LR_bicubic/X2'
RELATIVE_TRAINING_IMAGES_PATH_HR = './Dataset/DIV2K_train_HR'
RELATIVE_VALIDATION_IMAGES_PATH_LR = './Dataset/DIV2K_valid_LR_bicubic/X2'
RELATIVE_VALIDATION_IMAGES_PATH_HR = './Dataset/DIV2K_valid_HR'
IMAGE_FORMATS = ['png', 'jpg']
DATA_AUGMENTATION = True
AUGMENTATION_PROBABILITY = 0.5
MAXIMUM_PATCHES_IN_CACHE = 500  # MAXIMUM NUMBER OF PATCHES IN MEMORY
TOTAL_PATCH_NUMBERS = 7000


def return_number_of_pixels_in_dataset(image_directories_LR):
    print 'CALCULATING TOTAL NUMBER OF PIXELS IN THE DATASET'
    sum_pixels = 0
    for i, image_path in enumerate(tqdm(image_directories_LR)):
        image_lr = cv2.imread(image_path, 1)
        sum_pixels += image_lr.shape[0] * image_lr.shape[1]

    return sum_pixels


def return_all_images_directories(flag_lr_hr, flag_training_validation):
    images_directores = []
    if (flag_lr_hr == 'LR') and (flag_training_validation == 'training'):
        for image_format in IMAGE_FORMATS:
            images_directores += glob.glob(RELATIVE_TRAINING_IMAGES_PATH_LR + '/*.' + image_format)
    elif (flag_lr_hr == 'LR') and (flag_training_validation == 'validation'):
        for image_format in IMAGE_FORMATS:
            images_directores += glob.glob(RELATIVE_VALIDATION_IMAGES_PATH_LR + '/*.' + image_format)
    elif (flag_lr_hr == 'HR') and (flag_training_validation == 'training'):
        for image_format in IMAGE_FORMATS:
            images_directores += glob.glob(RELATIVE_TRAINING_IMAGES_PATH_HR + '/*.' + image_format)
    elif (flag_lr_hr == 'HR') and (flag_training_validation == 'validation'):
        for image_format in IMAGE_FORMATS:
            images_directores += glob.glob(RELATIVE_VALIDATION_IMAGES_PATH_HR + '/*.' + image_format)

    return images_directores


def find_index_in_HR(image_name, image_directories_HR):
    image_name_reduced = re.sub(r'x[2-9]', '', image_name)
    for index, image_path in enumerate(image_directories_HR):
        if re.search(r'.*{}.*'.format(image_name_reduced), image_path) is not None:
            return index


def augment_data(input_patch, output_patch):
    augmented_data_input, augmented_data_output = [], []

    if random.uniform(0, 1) < AUGMENTATION_PROBABILITY:
        augmented_input = cv2.flip(input_patch, flipCode=+1)
        augmented_output = cv2.flip(output_patch, flipCode=+1)
        augmented_data_input.append(augmented_input)
        augmented_data_output.append(augmented_output)

    if random.uniform(0, 1) < AUGMENTATION_PROBABILITY:
        augmented_input = cv2.flip(input_patch, flipCode=0)
        augmented_output = cv2.flip(output_patch, flipCode=0)
        augmented_data_input.append(augmented_input)
        augmented_data_output.append(augmented_output)

    if random.uniform(0, 1) < AUGMENTATION_PROBABILITY:
        augmented_input = imutils.rotate(input_patch, 90)
        augmented_output = imutils.rotate(output_patch, 90)
        augmented_data_input.append(augmented_input)
        augmented_data_output.append(augmented_output)


    return augmented_data_input, augmented_data_output

def process_image_and_add_to_the_cache(image_lr, image_hr, input_patches_cache, output_patches_cache, flag,
                                       total_number_of_pixels, total_saved_patches, flag_last_image=False):
    if flag == 'training':
        if flag_last_image:
            NUM_RANDOM_PATCHES_IN_ONE_IMAGE = int(TOTAL_PATCH_NUMBERS - total_saved_patches)
        else:
            NUM_RANDOM_PATCHES_IN_ONE_IMAGE = int(
                TOTAL_PATCH_NUMBERS * image_lr.shape[0] * image_lr.shape[1] / total_number_of_pixels)
        for i in range(NUM_RANDOM_PATCHES_IN_ONE_IMAGE):
            y = random.randint(0, image_lr.shape[0] - PATCH_SIZE - 1)
            x = random.randint(0, image_lr.shape[1] - PATCH_SIZE - 1)
            input_patch = np.zeros(shape=(PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS), dtype='uint8')
            image_patch_lr = image_lr[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            input_patch[0:image_patch_lr.shape[0], 0:image_patch_lr.shape[1]] = image_patch_lr

            output_patch = np.zeros(shape=(UPSCALE_FACTOR * PATCH_SIZE, UPSCALE_FACTOR * PATCH_SIZE, NUM_CHANNELS),
                                    dtype='uint8')
            image_patch_hr = image_hr[UPSCALE_FACTOR * y:UPSCALE_FACTOR * y + UPSCALE_FACTOR * PATCH_SIZE,
                             UPSCALE_FACTOR * x:UPSCALE_FACTOR * x + UPSCALE_FACTOR * PATCH_SIZE]
            output_patch[0:image_patch_hr.shape[0], 0:image_patch_hr.shape[1]] = image_patch_hr

            augmented_data_input, augmented_data_output = augment_data(input_patch, output_patch)

            input_patches_cache += augmented_data_input
            output_patches_cache += augmented_data_output

            input_patches_cache.append(input_patch)
            output_patches_cache.append(output_patch)

    # VALIDATION
    else:
        for y in range(0, image_lr.shape[0] - PATCH_SIZE - 1, PATCH_SIZE):
            for x in range(0, image_lr.shape[1] - PATCH_SIZE - 1, PATCH_SIZE):
                input_patch = np.zeros(shape=(PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS), dtype='uint8')
                image_patch_lr = image_lr[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                input_patch[0:image_patch_lr.shape[0], 0:image_patch_lr.shape[1]] = image_patch_lr

                output_patch = np.zeros(shape=(UPSCALE_FACTOR * PATCH_SIZE, UPSCALE_FACTOR * PATCH_SIZE, NUM_CHANNELS),
                                        dtype='uint8')
                image_patch_hr = image_hr[UPSCALE_FACTOR * y:UPSCALE_FACTOR * y + UPSCALE_FACTOR * PATCH_SIZE,
                                 UPSCALE_FACTOR * x:UPSCALE_FACTOR * x + UPSCALE_FACTOR * PATCH_SIZE]
                output_patch[0:image_patch_hr.shape[0], 0:image_patch_hr.shape[1]] = image_patch_hr

                input_patches_cache.append(input_patch)
                output_patches_cache.append(output_patch)

    return input_patches_cache, output_patches_cache


if __name__ == "__main__":

    total_saved_patches = 0
    for flag in ['training', 'validation']:

        database = Database('{}_h5'.format(flag), PATCH_SIZE, UPSCALE_FACTOR, NUM_CHANNELS)
        image_directories_LR = return_all_images_directories('LR', flag)
        image_directories_HR = return_all_images_directories('HR', flag)

        total_number_of_pixels = return_number_of_pixels_in_dataset(image_directories_LR)

        input_patches_cache = []
        output_patches_cache = []
        print 'Processing {} Dataset ...'.format(flag.upper())
        for i, image_path in enumerate(tqdm(image_directories_LR)):
            if i == (len(image_directories_LR) - 1):
                flag_last_image = True
            else:
                flag_last_image = False
            image_name = re.findall(r'.*/(.*)\..*$', image_path)[0]
            image_lr = cv2.imread(image_path, 1)
            image_hr = cv2.imread(image_directories_HR[find_index_in_HR(image_name, image_directories_HR)], 1)
            # split image into patches and put them in memory
            input_patches_cache, output_patches_cache = process_image_and_add_to_the_cache(image_lr, image_hr,
                                                                                           input_patches_cache,
                                                                                           output_patches_cache, flag,
                                                                                           total_number_of_pixels,
                                                                                           total_saved_patches,
                                                                                           flag_last_image)

            # save patches on disk and clean cache
            if len(input_patches_cache) >= MAXIMUM_PATCHES_IN_CACHE:
                total_saved_patches += len(input_patches_cache)
                database.append_to_h5_file(input_patches_cache, output_patches_cache)
                input_patches_cache = []
                output_patches_cache = []

        # save remaining patches in
        if len(input_patches_cache) != 0:
            database.append_to_h5_file(input_patches_cache, output_patches_cache)
            total_saved_patches += len(input_patches_cache)
        if flag == 'training':
            # get mean of the elements in database
            mean_input_image, mean_output_image = database.get_mean()
            cv2.imwrite('mean_input_image.png', mean_input_image)
            cv2.imwrite('mean_output_image.png', mean_output_image)

        del database
