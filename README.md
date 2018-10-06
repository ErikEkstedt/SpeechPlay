# Speech Play

In this repo I play around with wav-textword pairs. I intend to use soft attention over
the image from [Show, Attend and tell]() in order to correctly classify wich class of
words a spectrogram represents.

Data:
* 58252 training pairs
* The spectrograms are (99, 161) by default.
* 30 different classes

## So Far

* Dataset and Dataloaders loads wav from file and calculates spectrogram
* A Simple Spectrograms -> CNN -> classes model
* Simple training on cuda with tensorboard plotting loss


## Dataset

A nice dataset from kaggle with (text, wav) pairs.

Everything in dataset.py is based on 
[DadidS Kaggle Kernel](https://www.kaggle.com/davids1992/speech-representation-and-data-exploration)
