# twml_final
Trust Worthy Machine Learning (Spring 2023) Final Project

TIMNET (https://github.com/Jiaxin-Ye/TIM-Net_SER) with an added utily function `my_test`, which adapts the original `TIMNET_Model.test` method to arbitrary weights and datasets.

## CAVEATS:

1. Preprocess before training so that the training and the test data have the same input sizes. This can be done by padding them into same-sized (N, 39) MFCC arrays.


2. Train and test data must have the same class label set.


3. If input x.shape[0] is large (similar to that of IEMOCAP), must set dilation_size to 10 manually -- this is specified in the original code, so we're passing that on.


4. The results save path is defaulted to ./Results


5. Running my_test multiple times on the same dataset-weights pair will overwrite previous results.


## Instruction:

To run a test on a custom dataset, run

```
$ python main.py --mode my_test --data_path <data_path> --class_labels <class 1> ... <class k> --weights_path <weights_path>
```

or to run on a default dataset, run

```
$ python main.py --mode my_test --data <default_dataset_name> --weights_path <weights_path>
```
