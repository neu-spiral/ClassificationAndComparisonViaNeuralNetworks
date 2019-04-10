import numpy as np

'''
For each metric and alpha: 
1) find the best performance for each loss and compare these to choose the best loss function. 
2) find the best performance across all models and choose the corresponding lambda and lr as the best. 
'''

dir = '../../experiments/all_im/val'
################################
file_names = ['_abs_auc.txt', '_abs_acc.txt', '_abs_f1.txt', '_abs_prauc.txt',
              '_comp_auc.txt', '_comp_acc.txt', '_comp_f1.txt', '_comp_prauc.txt']
abs_losses = ['scaledCrossEntropy', 'scaledHinge']
comp_losses = ['scaledBTLoss', 'scaledThurstoneLoss', 'scaledCompCrossEntropy', 'scaledCompHinge']
# for each metric
for file_name in file_names:
    with open(dir + file_name, 'r+') as f:
        content = f.readlines()
    # get each line as a list
    content = [x.strip() for x in content]
    # remove white spaces
    content = [x for x in content if x != '']
    best_params = []
    # for each alpha
    for alpha in ['Alpha: 0.0', 'Alpha: 0.25', 'Alpha: 0.5', 'Alpha: 0.75', 'Alpha: 1.0']:
        best_perf_per_abs_loss = []
        best_perf_per_comp_loss = []
        # Find the best performance for each loss
        for abs_loss in abs_losses:
            values_str = np.array([content[line_ind + 1] for line_ind in range(len(content))
                                   if (alpha in content[line_ind]) and (abs_loss in content[line_ind])])
            values = np.array([float(elm) if '&' not in elm else 0 for elm in values_str])  # check if really is a float
            if values.size > 0:
                max_ind = np.argmax(values)
                best_perf_per_abs_loss.append(values[max_ind])
            else:
                best_perf_per_abs_loss.append(0)
        for comp_loss in comp_losses:
            values_str = np.array([content[line_ind + 1] for line_ind in range(len(content))
                                   if (alpha in content[line_ind]) and (comp_loss in content[line_ind])])
            values = np.array([float(elm) if '&' not in elm else 0 for elm in values_str])  # check if really is a float
            if values.size > 0:
                max_ind = np.argmax(values)
                best_perf_per_comp_loss.append(values[max_ind])
            else:
                best_perf_per_comp_loss.append(0)
        best_abs_loss_ind = int(np.argmax(np.array(best_perf_per_abs_loss)))
        best_comp_loss_ind = int(np.argmax(np.array(best_perf_per_comp_loss)))
        # Find the best lambda and learning rate with the given abs and comp losses
        params = np.array([line.split(' & Abs loss:')[0] for line in content if (alpha in line)])
        # get corresponding metric
        values_str = np.array(
            [content[line_ind + 1] for line_ind in range(len(content)) if (alpha in content[line_ind])])
        values = np.array([float(elm) if '&' not in elm else 0 for elm in values_str])  # check if really is a float
        max_ind = np.argmax(values)
        best_params.append(
            params[max_ind] + ' & ' + abs_losses[best_abs_loss_ind] + ' & ' + comp_losses[best_comp_loss_ind])
    with open(dir + '_best' + file_name, 'a') as best_f:
        for item in best_params:
            best_f.write('\n\n' + str(item))