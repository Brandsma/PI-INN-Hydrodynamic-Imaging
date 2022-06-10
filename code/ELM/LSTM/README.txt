Keras implementation of the LSTM. 

Run `python main.py' to run the LSTM network with default parametersettings (as used in the experiments). Different arguments can be supplied, but all of them have to be supplied simulateously. The arguments are (in this order):

    + n_nodes       -- The number of nodes
    + n_epochs      -- The number of training epochs
    + window_size   -- The size of each extracted window
    + Stride        -- The stride over the signal used to extract 
                       subsequent windows
    + Alpha         -- The learning rate
    + Decay         -- The learning rate decay. The learning rate decays 
                       after each (batch) update.
    + Data_split    -- How much of the dataset is used for training/    
                       testing. By default, a datasplit of 0.80 randomly 
                       assigns 80% of data to training, 10% to testing and 10 to validation. 
    + Dropout       -- How many of the LSTM cells' activation is not taken 
                       into account when calculating outputs. Reduces overfitting by decreasing reliance on particular nodes/features.
    + Train_loc     -- Path specifying the location of the dataset file.
    + Ac_fun        -- The interal LSTM activation function. 

An example of running this script would be: 

python main.py 20 30 15 2 0.05 1e-9 0.8 0 "../../data/preprocessed_data/Experiment_I/datased_used_in_experiments/all_ds64_zscore.mat" "relu"

If no arguments are supplied, the script uses the default arguments used during the experiments. Results are saved in the 'results/' subfolder.

For package/software dependencies, see the "Dependencies.txt" file one folder above this folder. 