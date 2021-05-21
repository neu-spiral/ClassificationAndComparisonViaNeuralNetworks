from __future__ import absolute_import
from __future__ import print_function
import pickle
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imresize
from os import listdir

class importData(object):
    def __init__(self, emotion1='happiness', emotion0='sadness', input_shape=(3, 224, 224), dir='./GIFGIF_DATA/'):
        self.emotion1 = emotion1
        self.emotion0 = emotion0
        self.input_shape = input_shape
        self.dir = dir

    def biLabels(self, labels):
        """
        This function will binarized labels.
        There are C classes {1,2,3,4,...,c} in the labels, the output would be c dimensional vector.
        Input:
            - labels: (N,) np array. The element value indicates the class index.
        Output:
            - biLabels: (N, C) array. Each row has and only has a 1, and the other elements are all zeros.
            - C: integer. The number of classes in the data.
        Example:
            The input labels = np.array([1,2,2,1,3])
            The binaried labels are np.array([[1,0,0],[0,1,0],[0,1,0],[1,0,0],[0,0,1]])
        """
        N = labels.shape[0]
        labels.astype(np.int)
        C = len(np.unique(labels))
        binarized = np.zeros((N, C))
        binarized[np.arange(N).astype(np.int), labels.astype(np.int).reshape((N,))] = 1
        return binarized, C

    def load_data(self, set):
        '''
        set: train/ val/ test
        all_abs_labels: (image_name, 1/0)
        all_comp_labels: (image1_name, image2_name, +1) is 1 > 2
        '''
        np.random.seed(1)
        # load training data matrices
        all_abs_labels = np.load(self.dir + set + '_abs_happy.npy')
        all_comp_labels = np.load(self.dir + set + '_comp_happy.npy')
        ###################
        # downsample training data
        #if set == 'train' and num_unique_images < all_abs_labels.shape[0]:
        #    all_abs_labels, all_comp_labels = self.sample_train_data(num_unique_images, all_abs_labels, all_comp_labels)
            ###################
        ###################
        # absolute images
        # load first image
        image_mtx = img_to_array(load_img(self.dir + 'labelled/' + all_abs_labels[0, 0])).astype(np.uint8)
        image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)
        abs_imgs = image_mtx[np.newaxis, :, :, :]
        # load images
        for row in np.arange(1, all_abs_labels.shape[0]):
            image_mtx = img_to_array(load_img(self.dir + 'labelled/' + all_abs_labels[row, 0])).astype(np.uint8)
            image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)[np.newaxis, :, :, :]
            abs_imgs = np.concatenate((abs_imgs, image_mtx), axis=0)
        # get corresponding labels
        if set == 'train':  # categorical due to softmax
            abs_labels, _ = self.biLabels(all_abs_labels[:, 1].astype(int))
        else:   # binary
            abs_labels = all_abs_labels[:, 1].astype(int)
        #####################
        # comparison images left
        # load first image
        image_mtx = img_to_array(load_img(self.dir + 'labelled/' + all_comp_labels[0, 0])).astype(np.uint8)
        image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)
        comp_imgs_1 = image_mtx[np.newaxis, :, :, :]
        # load images
        for row in np.arange(1, all_comp_labels.shape[0]):
            image_mtx = img_to_array(load_img(self.dir + 'labelled/' +  all_comp_labels[row, 0])).astype(np.uint8)
            image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)[np.newaxis, :, :, :]
            comp_imgs_1 = np.concatenate((comp_imgs_1, image_mtx), axis=0)
        # comparison images right
        # load first image
        image_mtx = img_to_array(load_img(self.dir + 'labelled/' + all_comp_labels[0, 1])).astype(np.uint8)
        image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)
        comp_imgs_2 = image_mtx[np.newaxis, :, :, :]
        # load images
        for row in np.arange(1, all_comp_labels.shape[0]):
            image_mtx = img_to_array(load_img(self.dir + 'labelled/' + all_comp_labels[row, 1])).astype(np.uint8)
            image_mtx = np.reshape(imresize(image_mtx, self.input_shape[1:]), self.input_shape)[np.newaxis, :, :, :]
            comp_imgs_2 = np.concatenate((comp_imgs_2, image_mtx), axis=0)
        # get corresponding labels
        comp_labels = all_comp_labels[:, 2].astype(int)

        return abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels

    def sample_train_data(self, num_unique_images, all_abs_labels, all_comp_labels):
        np.random.seed(1)
        # choose images
        abs_idx = np.random.permutation(np.arange(all_abs_labels.shape[0]))[:num_unique_images]
        # choose absolute labels
        new_abs_labels = all_abs_labels[abs_idx, :]
        new_imgs = new_abs_labels[:, 0]
        # choose comparison labels
        comp_idx = []
        for row_idx in range(all_comp_labels.shape[0]):
            # choose the comparison if the first or second image is in the absolute label set
            if all_comp_labels[row_idx, 0] in new_imgs or all_comp_labels[row_idx, 1] in new_imgs:
                comp_idx.append(row_idx)
        new_comp_labels = all_comp_labels[comp_idx, :]
        return new_abs_labels, new_comp_labels

    def create_partitions(self, valFold = 3, testFold = 4):
        file = 'gifgif-dataset-20150121-v1.csv'
        # Read all images in Happy and Sad folders
        imagenames1 = [f for f in listdir(self.dir + self.emotion1)]
        imagenames0 = [f for f in listdir(self.dir + self.emotion0)]
        image_name_list = imagenames1 + imagenames0
        # Choose folds
        np.random.seed(1)
        image_name_list = np.random.permutation(image_name_list)
        no_im_per_fold = int(len(image_name_list) / 5)
        image_name_list_by_fold = []
        for fold in range(5):
            image_name_list_by_fold.append(image_name_list[fold * no_im_per_fold:(fold + 1) * no_im_per_fold])
        #################################
        train_comp_labels = []
        val_comp_labels = []
        test_comp_labels = []
        train_abs_labels = []
        val_abs_labels = []
        test_abs_labels = []
        # get all absolute labels by fold
        # happy class, label 1
        for image_name in imagenames1:
            # test
            if image_name in image_name_list_by_fold[testFold]:
                test_abs_labels.append((image_name, 1))
            elif image_name in image_name_list_by_fold[valFold]:
                val_abs_labels.append((image_name, 1))
            # train
            else:
                train_abs_labels.append((image_name, 1))
        # sad class, label 0
        for image_name in imagenames0:
            # test
            if image_name in image_name_list_by_fold[testFold]:
                test_abs_labels.append((image_name, 0))
            elif image_name in image_name_list_by_fold[valFold]:
                val_abs_labels.append((image_name, 0))
            # train
            else:
                train_abs_labels.append((image_name, 0))
        ###############################
        # get all comparison labels by fold
        with open(self.dir + file) as f:
            next(f)  # First line is header.
            for line in f:
                emotion, image1_name, image2_name, choice = line.strip().split(",")
                if len(image1_name) == 0 or len(image2_name) == 0 or (
                            emotion != self.emotion1 and emotion != self.emotion0):
                    # Datum is corrupted, continue.
                    continue
                image1_name = image1_name + '.gif'
                image2_name = image2_name + '.gif'
                # test
                if image1_name in image_name_list_by_fold[testFold] and image2_name in image_name_list_by_fold[
                    testFold]:
                    if (choice == 'left' and emotion == self.emotion1) or (
                            choice == 'right' and emotion == self.emotion0):
                        test_comp_labels.append((image1_name, image2_name, +1))
                    elif (choice == 'right' and emotion == self.emotion1) or (
                            choice == 'left' and emotion == self.emotion0):
                        test_comp_labels.append((image1_name, image2_name, -1))
                # validation
                elif image1_name in image_name_list_by_fold[valFold] and image2_name in image_name_list_by_fold[
                    valFold]:
                    if (choice == 'left' and emotion == self.emotion1) or (
                            choice == 'right' and emotion == self.emotion0):
                        val_comp_labels.append((image1_name, image2_name, +1))
                    elif (choice == 'right' and emotion == self.emotion1) or (
                            choice == 'left' and emotion == self.emotion0):
                        val_comp_labels.append((image1_name, image2_name, -1))
                # train
                elif image1_name not in image_name_list_by_fold[valFold] and \
                    image2_name not in image_name_list_by_fold[valFold] and \
                    image1_name not in image_name_list_by_fold[testFold] and \
                    image2_name not in image_name_list_by_fold[testFold]:
                    if (choice == 'left' and emotion == self.emotion1) or (
                            choice == 'right' and emotion == self.emotion0):
                        train_comp_labels.append((image1_name, image2_name, +1))
                    elif (choice == 'right' and emotion == self.emotion1) or (
                            choice == 'left' and emotion == self.emotion0):
                        train_comp_labels.append((image1_name, image2_name, -1))
        ####################################
        train_abs_labels = np.array(train_abs_labels)
        val_abs_labels = np.array(val_abs_labels)
        test_abs_labels = np.array(test_abs_labels)
        train_comp_labels = np.array(train_comp_labels)
        val_comp_labels = np.array(val_comp_labels)
        test_comp_labels = np.array(test_comp_labels)
        np.save('train_abs_happy', train_abs_labels)
        np.save('val_abs_happy', val_abs_labels)
        np.save('test_abs_happy', test_abs_labels)
        np.save('train_comp_happy', train_comp_labels)
        np.save('val_comp_happy', val_comp_labels)
        np.save('test_comp_happy', test_comp_labels)