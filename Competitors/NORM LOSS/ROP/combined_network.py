from __future__ import absolute_import
from __future__ import print_function

from math import sqrt

import keras.backend as K
import numpy as np
import tensorflow as tf
from importData_rop import *
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import roc_auc_score

from googlenet_functional import *
'''
FOR NORMAL VS OTHERS
'''


def BTPred(scalars):
    """
    This is the output when we predict comparison labels. s1, s2 = scalars (beta.*x)
    """
    s1 = scalars[0]
    s2 = scalars[1]
    return s1 - s2
'''
def scaledThurstoneLoss(alpha):
    def ThurstoneLoss(y_true, y_pred):
        # P(si-sj|yij) = 0.5 * erfc(-yij*(si-sj) / sqrt(2))
        return - (1-alpha) * K.log(0.5 * tf.erfc(-y_true*y_pred / sqrt(2)))
    return ThurstoneLoss

def scaledBTLoss(alpha):
    def BTLoss(y_true, y_pred):
        """
        Negative log likelihood of Bradley-Terry Penalty, to be minimized. y = beta.*x
        y_true:-1 or 1
        y_pred:si-sj
        alpha: 0-1
        """
        exponent = K.exp(-y_true * (y_pred))
        return (1-alpha) * K.log(1 + exponent)
    return BTLoss
'''
def scaledCrossEntropy(alpha):
    def crossEntropy(y_true, y_pred):
        return alpha * K.categorical_crossentropy(y_true, y_pred)
    return crossEntropy

def scaledHinge(alpha):
    def hinge(y_true, y_pred):
        return alpha * K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)
    return hinge

'''
Comparison loss function from FACD paper
'''
def normPred(feature):
    """
    This is the output when we predict comparison labels. s1, s2 = scalars (beta.*x)
    """
    return tf.norm(feature, axis=1, keep_dims=True)**2

def scaledNormLoss(alpha):
    def normLoss(y_true, y_pred):
        '''max(||f_pos||^2 - ||f_neg||^2)'''
        return - (1-alpha) * y_true * y_pred
    return normLoss

class combined_deep_ROP(object):
    """
    Training and testing the neural network with 5000 images.
    """
    def __init__(self):
        self.data_gen = importData()

    def create_siamese(self, input_shape=(3, 224, 224), reg_param=0.0002, no_of_classes=3, no_of_score_layers=1,
                       max_no_of_nodes=128):
        input1 = Input(shape=input_shape)
        input2 = Input(shape=input_shape)
        # get features from base network
        feature1, feature2 = create_googlenet(input1, input2, reg_param=reg_param)
        # create final layers of absolute and comparison
        abs_out = Dense(no_of_classes, activation='softmax', kernel_regularizer=l2(reg_param), name='abs')
        comp_out = Dense(max_no_of_nodes, activation='relu', kernel_regularizer=l2(reg_param), name='comp')
        norm_layer = Lambda(normPred, output_shape=(1,), name='norm')
        # absolute part
        abs_out1 = abs_out(feature1)
        abs_net = Model(input1, abs_out1)
        # comparison part
        comp_out1 = comp_out(feature1)
        comp_out2 = comp_out(feature2)
        norm_out1 = norm_layer(comp_out1)
        norm_out2 = norm_layer(comp_out2)
        distance = Lambda(BTPred, output_shape=(1,))([norm_out1, norm_out2])
        comp_net = Model([input1, input2], distance)
        return abs_net, comp_net

    def train(self, save_model_name='./combined.h5', num_unique_images=80,
              reg_param=0.0002, no_of_classes=3, no_of_score_layers=1, max_no_of_nodes=128, learning_rate=1e-4,
              abs_loss=scaledCrossEntropy, comp_loss=scaledNormLoss, alpha=0.5, epochs=100, batch_size=32):
        """
        Training CNN except validation and test folds
        """
        abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels = \
                                                self.data_gen.load_training_data(num_unique_images=num_unique_images)
        abs_net, comp_net = self.create_siamese(reg_param=reg_param, no_of_classes=no_of_classes,
                                                no_of_score_layers=no_of_score_layers, max_no_of_nodes=max_no_of_nodes)
        # load imagenet weights
        abs_net.compile(loss=abs_loss(alpha=alpha), optimizer=SGD(learning_rate))
        abs_net.load_weights('./googlenet_weights.h5', by_name=True)
        comp_net.compile(loss=comp_loss(alpha=alpha), optimizer=SGD(learning_rate))
        # train on abs only, comp only or both
        for epoch in range(epochs):
            abs_net.fit(abs_imgs, abs_labels, batch_size=batch_size, epochs=1)
            comp_net.fit([comp_imgs_1, comp_imgs_2], comp_labels, batch_size=batch_size, epochs=1)
            print('**********End of epoch ' + str(epoch))
        # Save weights
        abs_net.save('Abs_' + save_model_name)
        comp_net.save('Comp_' + save_model_name)
        # with open('Hist_Comp_' + save_model_name, 'wb') as file_pi:
        #    pickle.dump(history.history, file_pi)

    def test(self, kthFold, model_file, num_unique_images=80,
              reg_param=0.0002, no_of_classes=3, no_of_score_layers=1, max_no_of_nodes=128, learning_rate=1e-4,
              abs_loss=scaledCrossEntropy, comp_loss=scaledNormLoss, alpha=0.5):
        """
        Testing CNN on validation/test fold.
        Predict 0th class, the class with the highest score
        """
        abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels = \
                                                self.data_gen.load_testing_data(kthFold, abs_thr='prep', test_set='100')
        abs_net, comp_net = self.create_siamese(reg_param=reg_param, no_of_classes=no_of_classes,
                                                no_of_score_layers=no_of_score_layers, max_no_of_nodes=max_no_of_nodes)
        # load weights and compile: BASE NETWORK HAS THE SAME WEIGHTS
        # make sure we have weights everytime we test. we have two single input-single output branches
        comp_test_model = Model(inputs=comp_net.input[0], outputs=comp_net.get_layer('norm').get_output_at(0))
        comp_test_model.load_weights('Comp_' + model_file, by_name=True)
        abs_net.load_weights('Abs_' + model_file, by_name=True)
        comp_net.load_weights('Comp_' + model_file, by_name=True)
        # compile all models
        comp_test_model.compile(loss=comp_loss(alpha=alpha), optimizer=SGD(learning_rate))
        abs_net.compile(loss=abs_loss(alpha=alpha), optimizer=SGD(learning_rate))
        comp_net.compile(loss=comp_loss(alpha=alpha), optimizer=SGD(learning_rate))
        #################TEST AUC, 0 is plus, 2 is normal
        if alpha == 0.0:  # only comparison training, use comp models
            abs_pred = 1 - comp_test_model.predict(abs_imgs)
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2])
        elif alpha == 1.0:  # only absolute training, use abs models
            abs_pred = abs_net.predict(abs_imgs)[:, 2]
            comp_pred = abs_net.predict(comp_imgs_1)[:, 0] - abs_net.predict(comp_imgs_2)[:, 0]
        else:
            abs_pred = abs_net.predict(abs_imgs)[:, 2]
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2])
        with open(str(kthFold) + '_auc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Num of im: ' + str(num_unique_images))
            file.write('\nAUC on absolute: ' + str(roc_auc_score(abs_labels, - abs_pred)))
            file.write('\nAUC on comparison: ' + str(roc_auc_score(comp_labels, comp_pred)))
        #################TEST OTHER METRICS
        if alpha == 0.0:  # only comparison training, use comp models
            abs_pred = comp_test_model.predict(abs_imgs)
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2])
            # a scalar output, classify with threshold 0.5
            abs_pred_thresholded = (abs_pred > 0.5).astype(int)
            comp_pred_thresholded = 2 * (comp_pred > 0.0).astype(int) - 1
        elif alpha == 1.0:  # only absolute training, use abs models
            abs_pred = abs_net.predict(abs_imgs)
            comp_pred = abs_net.predict(comp_imgs_1)[:, 0] - abs_net.predict(comp_imgs_2)[:, 0]
            # a 3 class output, take the maximum, 0 if plus
            abs_pred_012 = np.argmax(abs_pred, axis=1).astype(int)
            abs_pred_thresholded = (abs_pred_012 != 2).astype(int)
            comp_pred_thresholded = 2 * (comp_pred > 0.0).astype(int) - 1
        else:
            abs_pred = abs_net.predict(abs_imgs)
            comp_pred = comp_net.predict([comp_imgs_1, comp_imgs_2])
            # a 3 class output, take the maximum, 0 if plus
            abs_pred_012 = np.argmax(abs_pred, axis=1).astype(int)
            abs_pred_thresholded = (abs_pred_012 != 2).astype(int)
            comp_pred_thresholded = 2 * (comp_pred > 0.0).astype(int) - 1
        # calculate absolute metrics
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for l in range(abs_labels.shape[0]):
            if abs_pred_thresholded[l] == 1 and abs_labels[l] == 1:
                TP += 1
            elif abs_pred_thresholded[l] == 1 and abs_labels[l] == 0:
                FP += 1
            elif abs_pred_thresholded[l] == 0 and abs_labels[l] == 0:
                TN += 1
            else:
                FN += 1
        N_pos_test = TP + FN
        N_neg_test = FP + TN
        if (TP + FP) > 0:
            precision = 1.0 * TP / (TP + FP)
        else:
            precision = 0
        if (TP + FN) > 0:
            recall = 1.0 * TP / (TP + FN)
        else:
            recall = 0
        abs_accuracy = 1.0 * (TP + TN) / (N_pos_test + N_neg_test)
        if precision == 0 and recall == 0:
            abs_f1 = 0
        else:
            abs_f1 = 2.0 * precision * recall / (precision + recall)
        # calculate comparison metrics
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for l in range(comp_labels.shape[0]):
            if comp_pred_thresholded[l] == 1 and comp_labels[l] == 1:
                TP += 1
            elif comp_pred_thresholded[l] == 1 and comp_labels[l] == -1:
                FP += 1
            elif comp_pred_thresholded[l] == -1 and comp_labels[l] == -1:
                TN += 1
            else:
                FN += 1
        N_pos_test = TP + FN
        N_neg_test = FP + TN
        if (TP + FP) > 0:
            precision = 1.0 * TP / (TP + FP)
        else:
            precision = 0
        if (TP + FN) > 0:
            recall = 1.0 * TP / (TP + FN)
        else:
            recall = 0
        comp_accuracy = 1.0 * (TP + TN) / (N_pos_test + N_neg_test)
        if precision == 0 and recall == 0:
            comp_f1 = 0
        else:
            comp_f1 = 2.0 * precision * recall / (precision + recall)
        # save results
        with open(str(kthFold) + '_acc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Abs Loss: ' + str(abs_loss) + ' & Comp loss: ' + str(comp_loss))
            file.write('\nAccuracy on absolute: ' + str(abs_accuracy))
            file.write('\nAccuracy on comparison: ' + str(comp_accuracy))
        with open(str(kthFold) + '_f1.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Abs Loss: ' + str(abs_loss) + ' & Comp loss: ' + str(comp_loss))
            file.write('\nF1 score on absolute: ' + str(abs_f1))
            file.write('\nF1 score on comparison: ' + str(comp_f1))
        #################################################################

        # 5000
        abs_imgs, abs_labels, _, _, _ = self.data_gen.load_testing_data(kthFold, abs_thr='prep', test_set='5000')
        #################TEST AUC
        if alpha == 0.0:  # only comparison training, use comp models
            abs_pred = 1 - comp_test_model.predict(abs_imgs)
        elif alpha == 1.0:  # only absolute training, use abs models
            abs_pred = abs_net.predict(abs_imgs)[:, 2]  # take the plus output, needs scalar
        else:
            abs_pred = abs_net.predict(abs_imgs)[:, 2]
        with open('_auc_5000.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Num of im: ' + str(num_unique_images))
            file.write('\nAUC on absolute: ' + str(roc_auc_score(abs_labels, - abs_pred)))
        #################TEST OTHER METRICS
        # abs labels are -1 or 1 in this case
        if alpha == 0.0:  # only comparison training, use comp models
            abs_pred = comp_test_model.predict(abs_imgs)
            # a scalar output, classify with threshold 0.5
            abs_pred_thresholded = 2 * (abs_pred > 0.5).astype(int) - 1
        elif alpha == 1.0:  # only absolute training, use abs models
            abs_pred = abs_net.predict(abs_imgs)
            # a 3 class output, take the maximum, 0 if plus
            abs_pred_012 = np.argmax(abs_pred, axis=1).astype(int)
            abs_pred_thresholded = 2 * (abs_pred_012 != 2).astype(int) - 1
        else:
            abs_pred = abs_net.predict(abs_imgs)
            # a 3 class output, take the maximum, 0 if plus
            abs_pred_012 = np.argmax(abs_pred, axis=1).astype(int)
            abs_pred_thresholded = 2 * (abs_pred_012 != 2).astype(int) - 1
        # calculate metrics
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for l in range(abs_labels.shape[0]):
            if abs_pred_thresholded[l] == 1 and abs_labels[l] == 1:
                TP += 1
            elif abs_pred_thresholded[l] == 1 and abs_labels[l] == -1:
                FP += 1
            elif abs_pred_thresholded[l] == -1 and abs_labels[l] == -1:
                TN += 1
            else:
                FN += 1
        N_pos_test = TP + FN
        N_neg_test = FP + TN
        if (TP + FP) > 0:
            precision = 1.0 * TP / (TP + FP)
        else:
            precision = 0
        if (TP + FN) > 0:
            recall = 1.0 * TP / (TP + FN)
        else:
            recall = 0
        accuracy = 1.0 * (TP + TN) / (N_pos_test + N_neg_test)
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        with open('_acc_5000.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Num of im: ' + str(num_unique_images))
            file.write('\nAccuracy on absolute: ' + str(accuracy))
            file.write('\nN_pos: ' + str(N_pos_test) + ' & N_neg: ' + str(N_neg_test))
        with open('_f1_5000.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(reg_param) + ' & Learning rate: '
                       + str(learning_rate) + ' & Num of im: ' + str(num_unique_images))
            file.write('\nF1 score on absolute: ' + str(f1))
        





