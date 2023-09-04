# Codebase for "Cross-domain Transfer Learning and State Inference for Soft Robots via a Semi-supervised Sequential Variational Bayes Framework"

Authors: Shageenderan Sapai, Junn Yong Loo, Ze Yang Ding, Chee Pin Tan, Raphael C.-W. Phan, Vishnu Monn Baskaran, Surya Girinatha Nurzaman

Reference: @inproceedings{sapai2023cross,
  title={Cross-domain transfer learning and state inference for soft robots via a semi-supervised sequential variational bayes framework},
  author={Sapai, Shageenderan and Loo, Junn Yong and Ding, Ze Yang and Tan, Chee Pin and Phan, Rapha{\"e}l C-W and Baskaran, Vishnu Monn and Nurzaman, Surya Girinatha},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={552--559},
  year={2023},
  organization={IEEE}
}
 
Paper Link: https://ieeexplore.ieee.org/abstract/document/10160662

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


