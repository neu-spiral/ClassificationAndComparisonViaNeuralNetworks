import numpy as np

dir = '../experiments/less_im/val'
################################
file_names = ['_abs_auc.txt', '_abs_acc.txt', '_abs_f1.txt',
              '_comp_auc.txt', '_comp_acc.txt', '_comp_f1.txt']
# for each metric
for file_name in file_names:
    with open(dir + file_name, 'r+') as f:
        content = f.readlines()
    # get each line as a list
    content = [x.strip() for x in content]
    # remove white spaces
    content = [x for x in content if x != '']
    best_params = []
    # for different number of images
    for no_im in ['No tr im: 100', 'No tr im: 200', 'No tr im: 300', 'No tr im: 400',
                  'No tr im: 500', 'No tr im: 600', 'No tr im: 700']:
        # for each alpha
        for alpha in ['Alpha: 0.0', 'Alpha: 0.5', 'Alpha: 1.0']:
            params = np.array([line for line in content if (alpha in line and no_im in line)])
            if len(params) > 0:
                # get corresponding metric
                values = np.array([float(content[line_ind+1]) for line_ind in range(len(content))
                                  if (alpha in content[line_ind] and no_im in content[line_ind])])
                max_ind = np.argmax(values)
                best_params.append((params[max_ind], values[max_ind]))

    with open(dir + '_best' + file_name, 'a') as best_f:
        for item in best_params:
            best_f.write('\n\n' + str(item[0]))
            best_f.write('\n' + str(item[1]))