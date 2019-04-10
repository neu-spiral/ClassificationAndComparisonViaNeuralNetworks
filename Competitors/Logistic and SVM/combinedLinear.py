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

    def run_model(self, abs_features, abs_labels, comp_features, comp_labels,
                  train_type='LogLog', alpha=0.5, lamda=0.0002):
        # set: val/test
        # train
        if train_type == 'LogLog':
            beta, b = self.trainLogLog(abs_features, abs_labels, comp_features, comp_labels, alpha, lamda)
        elif train_type == 'LogSVM':
            beta, b = self.trainLogSVM(abs_features, abs_labels, comp_features, comp_labels, alpha, lamda)
        elif train_type == 'SVMLog':
            beta, b = self.trainSVMLog(abs_features, abs_labels, comp_features, comp_labels, alpha, lamda)
        else:
            beta, b = self.trainSVMSVM(abs_features, abs_labels, comp_features, comp_labels, alpha, lamda)
        return beta, b

    def test_model(self, beta, b, set, abs_features, abs_labels, comp_features, comp_labels,
                   train_type='LogLog', alpha=0.5, lamda=0.0002):
        # create predictions, both are between -1 and +1
        abs_pred = np.dot(abs_features, np.array(beta)) + b
        comp_pred = np.dot(comp_features, np.array(beta))
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
        with open(set+'_abs_auc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda) + ' & Train type: ' + str(train_type))
            file.write('\n' + str(roc_auc_score(abs_labels, abs_pred)))
        with open(set + '_comp_auc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda) + ' & Train type: ' + str(train_type))
            file.write('\n' + str(roc_auc_score(comp_labels, comp_pred)))
        with open(set + '_abs_acc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda) + ' & Train type: ' + str(train_type))
            file.write('\n' + str(abs_accuracy))
        with open(set + '_comp_acc.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda) + ' & Train type: ' + str(train_type))
            file.write('\n' + str(comp_accuracy))
        with open(set + '_abs_f1.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda) + ' & Train type: ' + str(train_type))
            file.write('\n' + str(abs_f1))
        with open(set + '_comp_f1.txt', 'a') as file:
            file.write('\n\nAlpha: ' + str(alpha) + ' & Lambda: ' + str(lamda) + ' & Train type: ' + str(train_type))
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