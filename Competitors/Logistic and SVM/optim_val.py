import numpy as np

set = '../rop/test prep/val'
################################
file_names = ['_abs_auc.txt', '_abs_acc.txt', '_abs_f1.txt',
              '_comp_auc.txt', '_comp_acc.txt', '_comp_f1.txt']
# for each metric
for file_name in file_names:
    with open(set + file_name, 'r+') as f:
        content = f.readlines()
    # get each line as a list
    content = [x.strip() for x in content]
    # remove white spaces
    content = [x for x in content if x != '']
    best_params = []
    # for each alpha
    for alpha in ['Alpha: 0.0', 'Alpha: 0.25', 'Alpha: 0.5', 'Alpha: 0.75', 'Alpha: 1.0']:
        params = np.array([line for line in content if alpha in line])
        if len(params) > 0:
            # get corresponding metric
            values = np.array([float(content[line_ind+1]) for line_ind in range(len(content)) if alpha in content[line_ind]])
            max_ind = np.argmax(values)
            best_params.append((params[max_ind], values[max_ind]))
    with open(set + '_best' + file_name, 'a') as best_f:
        for item in best_params:
            best_f.write('\n\n' + str(item[0]))
            best_f.write('\n' + str(item[1]))