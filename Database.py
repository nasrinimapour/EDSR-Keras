import os
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.utils import shuffle


class Database:
    def __init__(self, filename, patch_size, upscale_factor, num_channels):
        self.filename = filename
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.num_channels = num_channels

        ## MAKE DIRECTORY TO SAVE H5 DATASET
        if not os.path.isdir(self.filename):
            os.makedirs(self.filename)
        with h5py.File(self.filename + '/inputs.h5', 'w') as hf:
            hf.create_dataset('inputs', (1, self.patch_size, self.patch_size, self.num_channels),
                              maxshape=(None, self.patch_size, self.patch_size, self.num_channels), dtype='uint8',
                              chunks=True)
        with h5py.File(self.filename + '/outputs.h5', 'w') as hf:
            hf.create_dataset('outputs', (
                1, self.upscale_factor * self.patch_size, self.upscale_factor * self.patch_size, self.num_channels),
                              maxshape=(
                                  None, self.upscale_factor * self.patch_size, self.upscale_factor * self.patch_size,
                                  self.num_channels), dtype='uint8',
                              chunks=True)

    ## APPEND DATA TO EXISTING H5 DATASET
    def append_to_h5_file(self, input_patches, output_patches):
        input_patches_array = np.array(input_patches).reshape([-1, self.patch_size, self.patch_size, self.num_channels])
        output_patches_array = np.array(output_patches).reshape(
            [-1, self.upscale_factor * self.patch_size, self.upscale_factor * self.patch_size, self.num_channels])
        input_patches_array, output_patches_array = shuffle(input_patches_array, output_patches_array, random_state=0)
        with h5py.File(self.filename + '/inputs.h5', 'a') as hf:
            hf["inputs"].resize((hf["inputs"].shape[0] + input_patches_array.shape[0]), axis=0)
            hf["inputs"][-input_patches_array.shape[0]:] = input_patches_array

        with h5py.File(self.filename + '/outputs.h5', 'a') as hf:
            hf["outputs"].resize((hf["outputs"].shape[0] + output_patches_array.shape[0]), axis=0)
            hf["outputs"][-output_patches_array.shape[0]:] = output_patches_array

    def get_mean(self):
        print 'CALCULATING INPUT MEAN IMAGE ...'
        with h5py.File(self.filename + '/inputs.h5', 'a') as hf:
            mean_image_input = np.zeros(shape=(hf["inputs"].shape[1], hf["inputs"].shape[2], hf["inputs"].shape[3]),
                                        dtype='float32')
            for i in tqdm(range(hf["inputs"].shape[0])):
                image = np.array(hf["inputs"][i].astype('float32'))
                mean_image_input += (image / hf["inputs"].shape[0])

            mean_image_input = mean_image_input.astype('uint8')
        print 'CALCULATING OUTPUT MEAN IMAGE ...'
        with h5py.File(self.filename + '/outputs.h5', 'a') as hf:
            mean_image_output = np.zeros(shape=(hf["outputs"].shape[1], hf["outputs"].shape[2], hf["outputs"].shape[3]),
                                         dtype='float32')

            for i in tqdm(range(hf["outputs"].shape[0])):
                image = np.array(hf["outputs"][i]).astype('float32')
                mean_image_output += (image / hf["outputs"].shape[0])

            mean_image_output = mean_image_output.astype('uint8')
        return mean_image_input, mean_image_output

