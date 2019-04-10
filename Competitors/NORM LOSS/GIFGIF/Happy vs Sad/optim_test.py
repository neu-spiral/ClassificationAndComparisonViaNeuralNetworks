import numpy as np

no_im = 823
dir = '../experiments/less_im/'
################################
file_names = ['_abs_auc.txt', '_abs_acc.txt', '_abs_f1.txt',
              '_comp_auc.txt', '_comp_acc.txt', '_comp_f1.txt']
eval_models = []
# fetch the best model for each metric
for file_name in file_names:
    with open(dir + 'val_best' + file_name, 'r+') as f:
        content = f.readlines()
    # get each line as a list
    content = [x.strip() for x in content]
    # remove white spaces
    content = [x for x in content if x != '']
    best_params = np.array([line for line in content if 'No tr im: ' + str(no_im) in line])
    ####################################
    # test the best model for each alpha
    f.close()
    with open(dir + 'test' + file_name, 'r+') as f:
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
    eval_models.append(best_params[1])
    #eval_models.append(best_params[np.argmax(evals)])
##############
with open(dir + 'test_best_ ' + str(no_im) + '.txt', 'a') as best_f:
    best_f.write('\n\nBest model parameters:')
    for item in eval_models:
        best_f.write('\n' + str(item))
best_f.close()
# evaluate all models on each metric
for file_name in file_names:
    with open(dir + 'test' + file_name, 'r+') as f:
        content = f.readlines()
    # get each line as a list
    content = [x.strip() for x in content]
    # remove white spaces
    content = [x for x in content if x != '']
    evals = []
    for model in eval_models:
        if model in content:
            evals.append(float(content[content.index(model) + 1]))
    with open(dir + 'test_best_ ' + str(no_im) + '.txt', 'a') as best_f:
        best_f.write('\n\n' + file_name)
        for item in evals:
            best_f.write('\n' + str(item))
