import cvxpy as cp
import numpy as np
from sklearn.metrics import roc_auc_score

'''
absFeatures: N*d
absLabels: N*1, +1/-1
cmpFeatures: N*d - N*d
cmpLabels: N*1, +1/-1
'''

class combinedLinear(object):
    def __init__(self, input_shape=(3, 224, 224)):
        self.input_shape = input_shape

    def get_log_svm_predictions(self, dataset, abs_features, comp_features, alpha=0.5, lamda=0.0002):
        # get logistic predictions
        save_name = './' + dataset + '/logistic_models/' \
                                     'param_alpha_' + str(alpha) + '_lambda_' + str(lamda) + '_method_LogLog'
        beta = np.load(save_name + '.npy')
        b = beta[-1]
        beta = beta[:-1]
        abs_log_pred = np.dot(abs_features, np.array(beta)) + b
        comp_log_pred = np.dot(comp_features, np.array(beta))
        # get svm predictions
        save_name = './' + dataset + '/svm_models/' \
                                     'param_alpha_' + str(alpha) + '_lambda_' + str(lamda) + '_method_SVMSVM'
        beta = np.load(save_name + '.npy')
        b = beta[-1]
        beta = beta[:-1]
        abs_svm_pred = np.dot(abs_features, np.array(beta)) + b
        comp_svm_pred = np.dot(comp_features, np.array(beta))
        return abs_log_pred, comp_log_pred, abs_svm_pred, comp_svm_pred

    def run_ensemble(self, dataset, abs_features, abs_labels, comp_features, comp_labels, alpha=0.5, lamda=0.0002):
        # Log-SVM Prediction Features
        abs_log_pred, comp_log_pred, abs_svm_pred, comp_svm_pred = \
            self.get_log_svm_predictions(dataset, abs_features, comp_features, alpha, lamda)
        # concatenate predictions
        abs_pred_features = np.concatenate((abs_log_pred, abs_svm_pred), axis=1)
        comp_pred_features = np.concatenate((comp_log_pred, comp_svm_pred), axis=1)
        ################################################## train logistic on the predictions
        beta, b = self.trainLogLog(abs_pred_features, abs_labels, comp_pred_features, comp_labels, alpha, lamda)
        return beta, b

    def test_model(self, dataset, beta, b, set, abs_features, abs_labels, comp_features, comp_labels,
                   alpha=0.5, lamda=0.0002):
        # Log-SVM Prediction Features
        abs_log_pred, comp_log_pred, abs_svm_pred, comp_svm_pred = \
                self.get_log_svm_predictions(dataset, abs_features, comp_features, alpha, lamda)
        abs_pred_features = np.concatenate((abs_log_pred, abs_svm_pred), axis=1)
        comp_pred_features = np.concatenate((comp_log_pred, comp_svm_pred), axis=1)
        # create predictions, both are between -1 and +1
        abs_pred = np.dot(abs_pred_features, np.array(beta)) + b
        comp_pred = np.dot(comp_pred_features, np.array(beta))
        abs_pred_thresholded = 2 * (abs_pred > 0.0).astype(int) - 1
        comp_pred_thresholded = 2 * (comp_pred > 0.0).astype(int) - 1
        # calculate metrics
        # calculate absolute metrics
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
        with open('./' + dataset + '/' + set + '_abs_auc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda))
            file.write('\n' + str(roc_auc_score(abs_labels, abs_pred)))
        with open('./' + dataset + '/' + set + '_comp_auc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda))
            file.write('\n' + str(roc_auc_score(comp_labels, comp_pred)))
        with open('./' + dataset + '/' + set + '_abs_acc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda))
            file.write('\n' + str(abs_accuracy))
        with open('./' + dataset + '/' + set + '_comp_acc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda))
            file.write('\n' + str(comp_accuracy))
        with open('./' + dataset + '/' + set + '_abs_f1.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda))
            file.write('\n' + str(abs_f1))
        with open('./' + dataset + '/' + set + '_comp_f1.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda))
            file.write('\n' + str(comp_f1))

    def trainSVMSVM(self, absDataOrigin,absLabels,cmpDataOrigin,cmpLabels, absWeight, lamda):
        # The comparison data and label must be included.
        # Equation: min_{beta, const} alpha*sum(logisticLoss(absData))+(1-alpha)*sum(logisticLoss(cmpData))+lamda*norm(beta,1)
        # Parameter:
        # ------------
        # absDataOrigin : N by d numpy matrix where N the number of absolute label data and d is the dimension of data
        # abslabels : (N,) numpy array, +1 means positive label and -1 represents negative labels
        # cmpDataOrigin : N by d numpy matrix where N the number of comparion label data and d is the dimension of data
        # cmpLabels : (N,) numpy array, +1 means positive label and -1 represents negative labels
        # absWeight : the Weight on absolute label data. And (1-absWeight) would be the weight on comparison data.
        # lamda : weight on L1 penalty. Large lamda would have more zeros in beta.
        # normalizeWeight : binary value. 1 describes the normalize factor on  the absWeight and cmpWeight by its number of data.
        #                   0 shows no normalied factor happen.
        # Return:
        # ------------
        # beta : the logistic regression model parameter
        # const : the logistic regression global constant.
        cmpWeight = 1.0 - absWeight
        absN, d = np.shape(absDataOrigin)
        cmpN, _ = np.shape(cmpDataOrigin)
        beta = cp.Variable(d)
        const = cp.Variable(1)
        objective = absWeight * cp.sum_entries(cp.pos(1 - cp.mul_elemwise(absLabels, absDataOrigin * beta + const))) \
                    + cmpWeight * cp.sum_entries(cp.pos(1 - cp.mul_elemwise(cmpLabels, cmpDataOrigin * beta + const))) \
                    + lamda * cp.norm(beta, 2)
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve()
        return beta.value, const.value

    def trainLogSVM(self, absDataOrigin,absLabels,cmpDataOrigin,cmpLabels, absWeight, lamda):
        cmpWeight = 1.0 - absWeight
        absN, d = np.shape(absDataOrigin)
        cmpN, _ = np.shape(cmpDataOrigin)
        beta = cp.Variable(d)
        const = cp.Variable(1)
        objective = absWeight * cp.sum_entries(cp.logistic(cp.mul_elemwise(absLabels, absDataOrigin * beta + const)) \
                                               - cp.mul_elemwise(absLabels, absDataOrigin * beta + const)) + \
                    cmpWeight * cp.sum_entries(cp.pos(1 - cp.mul_elemwise(cmpLabels, cmpDataOrigin * beta + const))) \
                    + lamda * cp.norm(beta, 2)
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve(solver=cp.SCS)
        return beta.value, const.value

    def trainSVMLog(self, absDataOrigin,absLabels,cmpDataOrigin,cmpLabels, absWeight, lamda):
        cmpWeight = 1.0 - absWeight
        absN, d = np.shape(absDataOrigin)
        cmpN, _ = np.shape(cmpDataOrigin)
        beta = cp.Variable(d)
        const = cp.Variable(1)
        objective = cmpWeight * cp.sum_entries(cp.logistic(cp.mul_elemwise(cmpLabels, cmpDataOrigin * beta + const)) \
                                               - cp.mul_elemwise(cmpLabels, cmpDataOrigin * beta + const)) + \
                    absWeight * cp.sum_entries(cp.pos(1 - cp.mul_elemwise(absLabels, absDataOrigin * beta + const))) \
                    + lamda * cp.norm(beta, 2)
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve(solver=cp.SCS)
        return beta.value, const.value

    def trainLogLog(self, absDataOrigin,absLabels,cmpDataOrigin,cmpLabels, absWeight, lamda):
        cmpWeight = 1.0 - absWeight
        absN, d = np.shape(absDataOrigin)
        cmpN, _ = np.shape(cmpDataOrigin)
        beta = cp.Variable(d)
        const = cp.Variable(1)
        objective = absWeight * cp.sum_entries(cp.logistic(cp.mul_elemwise(absLabels, absDataOrigin * beta + const)) \
                                               - cp.mul_elemwise(absLabels, absDataOrigin * beta + const)) + \
                    cmpWeight * cp.sum_entries(cp.logistic(cp.mul_elemwise(cmpLabels, cmpDataOrigin * beta + const)) \
                                               - cp.mul_elemwise(cmpLabels, cmpDataOrigin * beta + const)) \
                    + lamda * cp.norm(beta, 2)
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve(solver=cp.SCS)
        return beta.value, const.value