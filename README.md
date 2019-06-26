# Classification and Comparison via Neural Networks
The code in this repository implements a combined neural network architecture that is trained on (and can be used to estimate) both class and pairwise comparison labels. Details can be found in the paper:
> Classification and Comparison via Neural Networks. Ilkay Yildiz, Peng Tian, Jennifer Dy, Deniz Erdogmus, James Brown, Jayashree Kalpathy-Cramer, Susan Ostmo, J. Peter Campbell, Michael F. Chiang, Stratis Ioannidis. Elsevier Journal of Neural Networks, Vol. 118, October 2019, pp. 65-80. https://doi.org/10.1016/j.neunet.2019.06.004

Implementation is in Python, using the Keras library.

The architecture is evaluated on 4 real-life datasets containing both class and comparison labels: GIFGIF Happiness, GIFGIF Pleasure, ROP, and FAC. The corresponding code files are in the folder named `Combined Neural Network`. Moreover, 4 competing methods combining both class and comparison labels are implemented: Logistic regression, SVM, ensemble of Logistic and SVM, and another deep learning approach. The corresponding code files are in the folder named `Competitors`. Finally, in the `Single Task Baseline` folder, a special case when the classification and comparison networks are identical is implemented.

From this point on, details about each file are provided:
* `combined_network.py` file contains the following functions:
    - `scaledCrossEntropy` and `scaledHinge` are classification losses, taking the trade-off parameter `alpha` as input.
    - `scaledBTLoss` and `scaledThurstoneLoss` are comparison losses, taking the trade-off parameter `alpha` as input.
    - `create_siamese` takes the regularization parameter `lambda`, number of layers following the base network, number of nodes in these layers as inputs, and creates the combined neural network. The base network GoogleNet is created by the files `googlenet_functional.py` and `googlenet_custom_layers.py`, containing the standard GoogleNet architecture implemented in Keras.  
    - `train` takes the regularization parameter `lambda`, number of layers following the base network, number of nodes in these layers, learning rate, loss functions, the trade-off parameter `alpha`, number of training epochs, and batch size as inputs. This function loads the training data, trains the architecture created by `create_siamese`, and saves the weights of absolute and comparison networks.
    - `test` takes the regularization parameter `lambda`, number of layers following the base network, number of nodes in these layers, learning rate, loss functions, and the trade-off parameter `alpha` as inputs. This function loads the test data, initializes the architecture created by `create_siamese` with the trained weights, and evaluates the architecture on AUC, accuracy, F1 score, and PRAUC of both class and comparison labels. Results are saved in `.txt` files.

* `main_combined_network.py` file takes `mode` (train or test), the trade-off parameter `alpha`, the regularization parameter `lambda`, learning rate, classification loss, comparison loss as arguments, and calls the `train` or `test` function in `combined_network.py` with respect to the chosen `mode`. If in test `mode`, the evaluation set can be set as `val` or `test`. Number of epochs, batch size, number of classes, directory of the dataset, and number of layers following the base network are also set in this main file. 

* `optim_val.py` reads the evaluation `.txt` files and finds the best performance over all training configurations on the validation set. Best model parameters are then saved into `.txt` files. `optim_test.py` reads the best model `.txt` files and evaluates the best models on the test set. 

# Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> Classification and Comparison via Neural Networks. Ilkay Yildiz, Peng Tian, Jennifer Dy, Deniz Erdogmus, James Brown, Jayashree Kalpathy-Cramer, Susan Ostmo, J. Peter Campbell, Michael F. Chiang, Stratis Ioannidis. Elsevier Journal of Neural Networks, Vol. 118, October 2019, pp. 65-80. https://doi.org/10.1016/j.neunet.2019.06.004

# Acknowledgements
Our work is supported by NIH (R01EY019474), NSF (SCH-1622542 at MGH; SCH-1622536 at Northeastern; SCH-1622679 at OHSU), and by unrestricted departmental funding from Research to Prevent Blindness (OHSU).
