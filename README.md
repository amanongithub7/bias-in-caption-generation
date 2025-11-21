# 🤨 Bias in Image Caption Generation Systems

This repository defines an Image Caption Generation deep learning model and contains scripts to compare the sentiment of generated captions against training data.

- `image-caption-generator` stores the encoder, decoder and attention layers of the model based on the architecture discussed in Xu et al.'s [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). 
- `train.py` trains the model based on global hyperparameters and then produces a sample caption. 
- `analysis/` stores some files used to deploy [Sheng et al.'s](https://arxiv.org/abs/1909.01326) sentiment and regard classifiers on our generated captions.

Project by Jiin Kim, Arjun Kallapur, Aman Oberoi. UCLA Winter 2022. 🐻
