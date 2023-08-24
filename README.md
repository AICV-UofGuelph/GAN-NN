  
# GAN-NN

**This repo (especially this readme) is being updated to make everything more clear, please check back often.**
This repository contains code for the NDM-GAN planner developed as part of my master's thesis project. The other planner developed is called S-LSTM (Stochastic-LSTM) and it's located [here](https://github.com/AICV-UofGuelph/LSTM-Autoencoder).







  

<!-- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -->

# gan_def_backups

  

GAN Type | Definition File | Training Notebook

:------: | :-------------: | :---------------:

AutoEncoder | GAN_AutoEnc.py | WGAN-GP.ipynb

Pix2Pix | GAN_P2P.py | Pix2Pix.ipynb

  
  
  

<!-- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -->

# WGAN-GP.ipynb

## Purpose

Configured for Gans in which the generator has the structure of an autoencoder with 3 input channels. Creates and trains a Wasserstein GAN with gradient penalty. Trains on pre-generated path data and frequently saves a checkpoint of the model's parameters. Records data about each run using Weights and Biases.

  

## Steps

1. Make sure any datasets you wish to use are present and follow the specified directory structure

2. Configure the training run using the provided constants

3. Run WGAN-GP.ipynb

  

## Constants

### I/O Config

-  ```RECORD_METRICS```: [Boolean] Whether or not the run will be logged on WandB and checkpoints saved

-  ```DATASET```: [String] The dataset to load

-  ```SUBSET```: [String] The subset to load (e.g. training, evaluation)

-  ```MAP_SHAPE```: [Tuple of ints] The dimensions of the map(s) being provided as input

  

### Run Configuration

-  ```NUM_EPOCHS```: [int] The number of epochs to train for

-  ```BATCH_SIZE```: [int] Number of items from the dataset in each batch

-  ```CRIT_ITERATIONS```: [int] How many times the critic will update its parameters for each update of the generator

-  ```LR_CRIT```: [Float] The critic's learning rate

-  ```LR_GEN```: [Float] The generator's learning rate

-  ```LAMBDA```: [Float or int] The coefficient of the gradient penalty term

  

### Critic Structure

-  ```NUM_LAYERS_CRIT```: [int] Number of layers in the critic

-  ```KERNEL_CRIT```: [List of ints] Kernel size for each layer of the critic

-  ```STRIDE_CRIT```: [List of ints] Stride for each layer of the critic

-  ```PAD_CRIT```: [List of ints] Padding for each layer of the critic

-  ```FEATURES_CRIT```: [List of ints] Number of input channels for each layer of the critic

  

### Generator Structure

-  ```NUM_LAYERS_GEN```: [int] Number of layers in the generator

-  ```KERNEL_GEN```: [List of ints] Kernel size for each layer of the generator

-  ```STRIDE_GEN```: [List of ints] Stride for each layer of the generator

-  ```PAD_GEN```: [List of ints] Padding for each layer of the generator

-  ```FEATURES_GEN```: [List of ints] Number of input channels for each layer of the generator

  
  

<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->

#

## GAN.py: Stores GAN class definition so it can be easily loaded later.

  
  
  

<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->

# eval_GAN.ipynb

  

## Purpose

Loads a trained GAN and evaluates it on the specified dataset. Prints a few example outputs.

  

## Steps

1. Train a model using WGAN-GP.ipynb

2. Set the desired evaluation set using the provided constants

3. Run eval_GAN.ipynb

  

## Changing Variables

-  ```DATASET```: [String] The dataset to pass as input to the GAN

-  ```SUBSET```: [String] The subset to pass as input to the GAN

-  ```RUN_NAME```: [String] The run corresponding to the model being loaded

-  ```STEP```: [int] The step from which to load the model's parameters

-  ```BATCH_SIZE```: [int] Number of items in each batch

-  ```SMOOTH_VAL```: [int] Number of times to smooth the output path

-  ```NUM_SAMPLES```: [int] Number of example outputs to display

-  ```FIG_SCALE```: [int] Desired number of pixels per inch in the displayed outputs

  
  

<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->

# data_manager.py

  

## Purpose

A module containing methods for saving/loading data associated with the GAN. Functions include loading datasets, loading maps (before or after SDF), saving model parameters, etc.

  

## Steps

1. Import into your python script

2. Use the appropriate function to retrieve or store data

  

## Constants

-  ```INPUTS_DIR```: [String] The name of the folder to use as the root directory when searching for input

-  ```CHKPTS_DIR```: [String] The name of the folder to use as the root directory when saving GAN definitions and checkpoints

# Authors and Acknowledgement
This project was only possible thanks to the help from undergraduate research assistants Aidan Holvik and Rachael Mohl.