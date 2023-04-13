# Codebase for "Cross-domain Transfer Learning and State Inference for Soft Robots via a Semi-supervised Sequential Variational Bayes Framework"

Authors: Shageenderan Sapai, Junn Yong Loo, Ze Yang Ding, Chee Pin Tan, Raphael C.-W. Phan, Vishnu Monn Baskaran, Surya Girinatha Nurzaman

Reference: 
 
Paper Link: 

Contact: shageenderan.sapai@monash.edu, loo.junnyong@monash.edu

This directory contains implementations of the DSVB framework for transfer learning and state inference for soft robots demonstrated on a Pneumatic Soft Finger(PSF) platform.

To run the pipeline for training and evaluation on DSVB framwork, simply run 
python3 -m [ADD HERE].

### Code explanation

(1) VRNN_DAT.py
- Tensorflow implementation of DSVB

(2) Weights directory  
- Contains the weights of the model after training

(3) Data directory  
- Contains the PSF dataset

(4) main_dsvb.py
- [ADD HERE]

### Command inputs:

-   data_name: sine, stock, or energy
-   seq_len: sequence length
-   module: gru, lstm, or lstmLN
-   hidden_dim: hidden dimensions
-   num_layers: number of layers
-   iterations: number of training iterations
-   batch_size: the number of samples in each batch
-   metric_iterations: number of iterations for metric computation

Note that network parameters should be optimized for different datasets.

### Example command

```shell
$ python3 main_timegan.py --data_name stock --seq_len 24 --module gru
--hidden_dim 24 --num_layer 3 --iteration 50000 --batch_size 128 
--metric_iteration 10
```

### Outputs


