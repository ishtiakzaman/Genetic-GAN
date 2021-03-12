from __future__ import division
import numpy as np
import h5py

class DatasetHDF5:
    def __init__(self, dataset_name, batch_size, shuffle=True, split=85, feature_name_list=['image','label']):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training_iter = 0
        self.validation_iter = 0
        self.split = split
        self.file = h5py.File(dataset_name, 'r')
        self.feature_name_list = feature_name_list
        self.dataset_features = [self.file[feature_name] for feature_name in feature_name_list]
        self.num_examples = self.dataset_features[0].shape[0]
        self.num_training = np.int(self.num_examples * split / 100)
        self.num_validation = self.num_examples - self.num_training
        self.__initialize__()
        print('Dataset for', self.num_examples, 'examples,', self.batch_size, 'batch size,', self.num_training, 'training data,', \
            self.num_validation, 'validation data.')

    def __initialize__(self):
        self.num_validation_iter_per_epoch = np.int(np.ceil(self.num_validation / self.batch_size))
        if self.num_training % self.batch_size == 0:
            self.num_training_iter_per_epoch = self.num_training // self.batch_size
        else:
            self.num_training_iter_per_epoch = self.num_training // self.batch_size + 1
        if self.num_validation % self.batch_size == 0:
            self.num_validation_iter_per_epoch = self.num_validation // self.batch_size
        else:
            self.num_validation_iter_per_epoch = self.num_validation // self.batch_size + 1
        if self.shuffle == True:
            self.index = np.arange(self.num_training_iter_per_epoch)
            np.random.shuffle(self.index)
        assert self.num_training >= self.batch_size and (self.split == 100 or self.num_validation >= self.batch_size)

    def get_length(self):
        return self.num_examples

    def reset(self, batch_size=None):
        self.training_iter = 0
        self.validation_iter = 0
        print('Dataset Reset',)
        if batch_size is not None:
            self.batch_size = batch_size
            self.__initialize__()
            print('with batch size', batch_size,)
        print

    def load_batch(self, mode):
        assert mode in ['train', 'validation']
        epoch_happened = False
        if mode == 'train':
            if self.shuffle == True:
                start_index = (self.index[self.training_iter] * self.batch_size) % self.num_training 
                end_index = ((self.index[self.training_iter] + 1) * self.batch_size) % self.num_training
            else:
                start_index = (self.training_iter * self.batch_size) % self.num_training 
                end_index = ((self.training_iter + 1) * self.batch_size) % self.num_training
            if start_index >= end_index:
                end_index = self.num_training
            self.training_iter = self.training_iter + 1
            if self.training_iter == self.num_training_iter_per_epoch:
                epoch_happened = True
                if self.shuffle == True:
                    np.random.shuffle(self.index)
                self.training_iter = 0
        else:
            start_index = (self.validation_iter * self.batch_size) % self.num_validation + self.num_training
            end_index = ((self.validation_iter + 1) * self.batch_size) % self.num_validation + self.num_training
            if start_index > end_index:
                end_index = self.num_examples
            self.validation_iter = self.validation_iter + 1
            if self.validation_iter == self.num_validation_iter_per_epoch:
                epoch_happened = True
                self.validation_iter = 0

        features = [self.dataset_features[i][start_index:end_index] for i in range(len(self.feature_name_list))]
        features.insert(0, epoch_happened)

        return tuple(features)

