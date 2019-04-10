import argparse

from combined_network import *

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='NN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, default='reg', choices=['train', 'test'],
                        help='choose training the neural network or evaluating it.')
    parser.add_argument('alpha', type=float)
    parser.add_argument('reg_param', type=float)
    parser.add_argument('lr', type=float)
    args = parser.parse_args()
    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    kthFold = 3 #validation fold:3, test fold:0, train with 1,2,4
    num_unique_images = 60
    reg_param = args.reg_param
    learning_rate = args.lr
    alpha = args.alpha
    #######
    abs_loss = scaledCrossEntropy
    comp_loss = scaledNormLoss
    no_of_classes = 3
    epochs = 50
    batch_size = 24
    # no_of_score_layers = 1
    max_no_of_nodes = 128  # no of fused features
    save_model_name = 'model_alpha_' + str(alpha) + '_lambda_' + str(reg_param) + '_lr_' + str(learning_rate)
    #######################################################################################################
    if args.mode == 'train':
        combined_net = combined_deep_ROP()
        combined_net.train(save_model_name=save_model_name, num_unique_images=num_unique_images,
              reg_param=reg_param, no_of_classes=no_of_classes, max_no_of_nodes=max_no_of_nodes, learning_rate=learning_rate,
              abs_loss=abs_loss, comp_loss=comp_loss, alpha=alpha, epochs=epochs, batch_size=batch_size)
    elif args.mode == 'test':
        combined_net = combined_deep_ROP()
        combined_net.test(kthFold, save_model_name, num_unique_images=num_unique_images,
              reg_param=reg_param, no_of_classes=no_of_classes,  max_no_of_nodes=max_no_of_nodes, learning_rate=learning_rate,
              abs_loss=abs_loss, comp_loss=comp_loss, alpha=alpha)
    else:
        print('mode must be either train or test')
