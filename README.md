
# Char-RNN using PyTorch

This project implements a Recurrent Neural Network (RNN) model for text generation using PyTorch. Repository-based [spro/char-rnn.pytorch](https://github.com/spro/char-rnn.pytorch).

## Overview

The RNN model is trained to predict the next character in a text string. After training, the model can generate new texts based on an initial text (prime string).

## Requirements

- Python 3.6+
- PyTorch
- tqdm

## Project Structure

- `training_module.py`: Script to train the RNN model.
- `text_generator.py`: Script to generate text using a trained model.
- `helpers.py`: Auxiliary functions for text processing.
- `model.py`: Definition of the RNN model.

## Use

### Model Training

To train the model, use the training_module.py script with the following parameters:

```
usage: training_module.py [-h] [--model MODEL] [--n_epochs N_EPOCHS]
                		  [--print_every PRINT_EVERY] [--hidden_size HIDDEN_SIZE]
                  		  [--n_layers N_LAYERS] [--learning_rate LEARNING_RATE]
                		  [--chunk_len CHUNK_LEN] [--batch_size BATCH_SIZE] [--shuffle]
                		  [--cuda]
                		  filename

positional arguments:
  filename              Name of the training text file.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         RNN model type (default: "gru")
  --n_epochs N_EPOCHS   Number of training epochs (default: 2000)
  --print_every PRINT_EVERY
                        Interval for printing progress (default: 100)
  --hidden_size HIDDEN_SIZE
                        Hidden layer size (default: 100)
  --n_layers N_LAYERS   Number of layers (default: 2)
  --learning_rate LEARNING_RATE
                        Learning Rate (default: 0.01)
  --chunk_len CHUNK_LEN
                        Text segment length for training (default: 200)
  --batch_size BATCH_SIZE
                        Size of batch (default: 100)
  --shuffle             Shuffle of dataset
  --cuda                Use CUDA (GPU) for training
```

Example of use:

```sh
python train.py input.txt --model gru --n_epochs 5000 --cuda
```

### Text Generation

To generate text using a trained model, use the `generate.py` script with the following parameters:

```
usage: generate.py [-h] [-p PRIME_STR] [-l PREDICT_LEN] [-t TEMPERATURE]
                   [--cuda]
                   filename

positional arguments:
  filename              File name of the trained model.

optional arguments:
  -h, --help            show this help message and exit
  -p PRIME_STR, --prime_str PRIME_STR
                        Initial string for text generation (default: "A")
  -l PREDICT_LEN, --predict_len PREDICT_LEN
                        Length of text to be generated (default: 100)
  -t TEMPERATURE, --temperature TEMPERATURE
                        Sample temperature (default: 0.8)
  --cuda                Use CUDA (GPU) for text generation
```

Example of use:

```sh
python generate.py model.pt --prime_str "Era uma vez" --predict_len 200 --cuda
```

## Créditos

This project is based on work available in [spro/char-rnn.pytorch](https://github.com/spro/char-rnn.pytorch).
