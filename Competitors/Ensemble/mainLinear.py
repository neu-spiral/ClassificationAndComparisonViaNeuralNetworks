import argparse
from combinedLinear import *

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='logistic', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('alpha', type=float)
    parser.add_argument('lamda', type=float)
    args = parser.parse_args()
    #######################################################################################################
    # Setting parameters
    #######################################################################################################
    dataset = args.dataset
    alpha = args.alpha
    lamda = args.lamda
    save_beta = './ens_models/' + dataset + '_alpha_' + str(alpha) + '_lambda_' + str(lamda)
    # Data related
    abs_features = np.load('./_features/abs_feat_' + str(args.mode) + '_' + dataset + '.npy')
    abs_labels = np.load('./_features/abs_label_' + str(args.mode) + '_' + dataset + '.npy')
    comp_features = np.load('./_features/comp_feat_' + str(args.mode) + '_' + dataset + '.npy')
    comp_labels = np.load('./_features/comp_label_' + str(args.mode) + '_' + dataset + '.npy')
    #######################################################################################################
    combined = combinedLinear()
    if args.mode == 'train':
        beta, b = combined.run_ensemble(dataset, abs_features, abs_labels, comp_features, comp_labels,
                  alpha=alpha, lamda=lamda)
        np.save(save_beta, np.concatenate((np.array(beta), np.array([b])[:,np.newaxis])))
    else:
        beta = np.load(save_beta + '.npy')
        b = beta[-1]
        beta = beta[:-1]
        combined.test_model(dataset, beta, b, args.mode, abs_features, abs_labels, comp_features, comp_labels,
                   alpha=alpha, lamda=lamda)

