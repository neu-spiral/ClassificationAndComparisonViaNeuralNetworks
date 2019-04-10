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
    set = 'val'
    epochs = 75
    batch_size = 8
    reg_param = args.reg_param
    learning_rate = args.lr
    alpha = args.alpha
    ###
    abs_loss = scaledCrossEntropy
    comp_loss = scaledNormLoss
    input_shape = (3, 224, 224)
    no_of_classes = 2
    dir = "./GIFGIF_DATA/"
    # no_of_score_layers = 1
    max_no_of_nodes = 128  # no of fused features
    save_model_name = 'model_alpha_' + str(alpha) + '_lambda_' + str(reg_param) + '_lr_' + str(learning_rate)
    #######################################################################################################
    if args.mode == 'train':
        combined_net = combined_deep_ROP(input_shape=input_shape, no_of_classes=no_of_classes, dir=dir)
        combined_net.train(save_model_name=save_model_name,
                           reg_param=reg_param, max_no_of_nodes=max_no_of_nodes, learning_rate=learning_rate,
                           abs_loss=abs_loss, comp_loss=comp_loss, alpha=alpha, epochs=epochs, batch_size=batch_size)
    elif args.mode == 'test':
        combined_net = combined_deep_ROP(input_shape=input_shape, no_of_classes=no_of_classes, dir=dir)
        combined_net.test(set, save_model_name,
                          reg_param=reg_param, max_no_of_nodes=max_no_of_nodes, learning_rate=learning_rate,
                          abs_loss=abs_loss, comp_loss=comp_loss, alpha=alpha)
    else:
        print('mode must be either train or test')
