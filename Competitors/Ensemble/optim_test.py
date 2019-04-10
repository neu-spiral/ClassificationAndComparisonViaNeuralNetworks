import numpy as np
from math import ceil

set = './fac/'
nPosAbs = 316
nNegAbs = 388
nPosComp = 107
nNegComp = 117
################################
file_names = ['_abs_auc.txt', '_abs_acc.txt', '_abs_f1.txt',
              '_comp_auc.txt', '_comp_acc.txt', '_comp_f1.txt']
eval_models = []
# fetch the best model for each metric
for file_name in file_names:
    with open(set + 'val_best' + file_name, 'r+') as f:
        content = f.readlines()
    # get each line as a list
    content = [x.strip() for x in content]
    # remove white spaces
    content = [x for x in content if x != '']
    # !!!!!!!!!!!!!!!!
    best_params = np.array([line for line in content if 'Alpha:' in line])
    ####################################
    # test the best model for each alpha
    f.close()
    with open(set + 'test' + file_name, 'r+') as f:
        content = f.readlines()
    # get each line as a list
    content = [x.strip() for x in content]
    # remove white spaces
    content = [x for x in content if x != '']
    evals = []
    for param in best_params:
        if param in content:
            evals.append(float(content[content.index(param) + 1]))
    # evaluate on alpha=0, best test alpha and alpha=1
    eval_models.append(best_params[0])
    eval_models.append(best_params[-1])
    eval_models.append(best_params[np.argmax(evals)])
##############
with open(set + 'test_best.txt', 'a') as best_f:
    best_f.write('\n\nBest model parameters:')
    for item in eval_models:
        best_f.write('\n' + str(item))
best_f.close()
# evaluate all models on each metric
for file_name in file_names:
    with open(set + 'test' + file_name, 'r+') as f:
        content = f.readlines()
    # get each line as a list
    content = [x.strip() for x in content]
    # remove white spaces
    content = [x for x in content if x != '']
    evals = []
    for model in eval_models:
        metric_val = float(content[content.index(model) + 1])
        # calculate confidence interval
        if 'auc' in file_name:
            pxxy = metric_val / (2 - metric_val)
            pxyy = (2 * metric_val ** 2) / (1 + metric_val)
            if 'abs' in file_name:
                SE = 1.96 * np.sqrt((metric_val * (1 - metric_val) + (nPosAbs - 1) * (pxxy - metric_val ** 2)
                                     + (nNegAbs - 1) * (pxyy - metric_val ** 2)) / (nPosAbs * nNegAbs))
            else:
                SE = 1.96 * np.sqrt((metric_val * (1 - metric_val) + (nPosComp - 1) * (pxxy - metric_val ** 2)
                                     + (nNegComp - 1) * (pxyy - metric_val ** 2)) / (nPosComp * nNegComp))
        else:
            if 'abs' in file_name:
                SE = 1.96 * np.sqrt(metric_val * (1 - metric_val) / (nPosAbs + nNegAbs))
            else:
                SE = 1.96 * np.sqrt(metric_val * (1 - metric_val) / (nPosComp + nNegComp))
        # three decimal points
        metric_val = ceil(metric_val * 1000) / 1000
        SE = ceil(SE * 1000) / 1000
        evals.append('$' + str(metric_val) + ' \pm ' + str(SE) + '$')
    with open(set + 'test_best.txt', 'a') as best_f:
        best_f.write('\n\n' + file_name)
        for item in evals:
            best_f.write('\n' + str(item))
