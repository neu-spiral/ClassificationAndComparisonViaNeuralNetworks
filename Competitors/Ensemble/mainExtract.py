import argparse
import numpy as np
from keras.layers import Input
from keras.models import Model
from googlenet_functional import *
from importData_rop import *

'''
absFeatures: N*d
absLabels: N*1, +1/-1
cmpFeatures: N*d - N*d
cmpLabels: N*1, +1/-1
'''

def extract_features(abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels):
    # create base network
    input1 = Input(shape=(3,224,224))
    input2 = Input(shape=(3,224,224))
    feature1, _ = create_googlenet(input1, input2)
    base_net = Model(input1, feature1)
    base_net.load_weights('./googlenet_weights.h5', by_name=True)
    base_net.compile(loss='mean_squared_error', optimizer='sgd')
    # extract features
    abs_features = base_net.predict(abs_imgs)
    comp_features_1 = base_net.predict(comp_imgs_1)
    comp_features_2 = base_net.predict(comp_imgs_2)
    comp_features = comp_features_1 - comp_features_2
    # labels between +1 and -1
    abs_labels = 2 * abs_labels - 1
    return abs_features, abs_labels, comp_features, comp_labels

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='logistic', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str)
    parser.add_argument('abs_thr', type=str)
    args = parser.parse_args()
    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    set = args.mode
    abs_thr = args.abs_thr
    save_name = str(set) + '_rop_' + str(abs_thr)
    # Data related
    importer = importData()
    if set == 'train':
        abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels = importer.load_training_data(num_unique_images=60)
        # make plus 1
        temp = np.zeros((abs_labels.shape[0],))
        temp[np.where(abs_labels[:, 0] == 1)[0]] = 1
        if abs_thr == 'prep':
            temp[np.where(abs_labels[:, 1] == 1)[0]] = 1
        abs_labels = temp.astype(int)
    elif set == 'val':
        abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels = importer.load_testing_data(3, abs_thr=abs_thr, test_set='100')
    else:
        abs_imgs, abs_labels, _, _, _ = importer.load_testing_data(0, abs_thr=abs_thr, test_set='5000')
        _, _, comp_imgs_1, comp_imgs_2, comp_labels = importer.load_testing_data(0, abs_thr=abs_thr,
                                                                                        test_set='100')

    abs_features, abs_labels, comp_features, comp_labels = extract_features\
                            (abs_imgs, abs_labels, comp_imgs_1, comp_imgs_2, comp_labels)
    np.save('abs_label_' + save_name, abs_labels)
