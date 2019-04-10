import argparse

from combined_network import *

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='NN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, default='reg', choices=['train', 'test'],
                        help='choose training the neural network or evaluating it.')
    parser.add_argument('alpha', type=float)
    parser.add_argument('reg_param', type=float)
    parser.add_argument('lr', type=float)
    parser.add_argument('abs_loss', type=str)
    parser.add_argument('comp_loss', type=str)
    args = parser.parse_args()
    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    kthFold = 3 #validation fold:3, test fold:0, train with 1,2,4
    epochs = 50
    batch_size = 24
    reg_param = args.reg_param
    learning_rate = args.lr
    alpha = args.alpha
    if args.abs_loss == 'cat':
        abs_loss = scaledCrossEntropy
    else:
        abs_loss = scaledHinge
    if args.comp_loss == 'cat':
        comp_loss = scaledCompCrossEntropy
    elif args.comp_loss == 'hinge':
        comp_loss = scaledCompHinge
    elif args.comp_loss == 'BT':
        comp_loss = scaledBTLoss
    else:
        comp_loss = scaledThurstoneLoss
    #######
    no_of_classes = 3
    no_of_score_layers = 1
    # max_no_of_nodes = 128
    save_model_name = 'model_alpha_' + str(alpha) + '_lambda_' + str(reg_param) + '_lr_' + str(learning_rate) + \
                      '_' + str(args.abs_loss) + '_' + str(args.comp_loss)
    #######################################################################################################
    if args.mode == 'train':
        combined_net = combined_deep_ROP()
        combined_net.train(save_model_name=save_model_name,
              reg_param=reg_param, no_of_classes=no_of_classes, no_of_score_layers=no_of_score_layers, learning_rate=learning_rate,
              abs_loss=abs_loss, comp_loss=comp_loss, alpha=alpha, epochs=epochs, batch_size=batch_size)
    elif args.mode == 'test':
        combined_net = combined_deep_ROP()
        combined_net.test(kthFold, save_model_name,
              reg_param=reg_param, no_of_classes=no_of_classes, no_of_score_layers=no_of_score_layers, learning_rate=learning_rate,
              abs_loss=abs_loss, comp_loss=comp_loss, alpha=alpha)
    else:
        print('mode must be either train or test')
