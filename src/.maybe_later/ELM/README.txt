Tensorflow implementation of OS-ELM. Adapted from https://github.com/otenim/TensorFlow-OS-ELM. 

Run `python train.py n_nodes window_size stride' where
    - n_nodes is an integer giving the number of hidden nodes
    - window_size is an integer specifying the window size
    - stride is an integer specifying the stride over the signal

If no arguments are supplied, the script uses the default arguments used during the experiments: 30000 nodes, a window size of 15 and a stride of 1. 

Results are saved in the 'results/' subfolder.

For package/software dependencies, see the "Dependencies.txt" file one folder above this folder. 