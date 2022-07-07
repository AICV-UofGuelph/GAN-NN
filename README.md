# create_paths.ipynb

## Purpose

Takes path data output from GAN generators and sorts/smoothens it into path data that can be used to create waypoints.

<!-- input/output pictures -->

## Steps

1. Put the map file you want to use in the 'env/' folder, update ```MAP_NAME``` variable accordingly (see next section).
2. Start Python 3.9 kernel.
3. Run file.

## Changing Variables

Go to start of cell 2 to alter:
- ```MAP_NAME```: Map file name (without extension).



# multiple_condition_GAN.ipynb

## Purpose

<!-- short description of what program does -->

<!-- input/output pictures -->

## Steps

<!-- 1. Describe setup -->
<!-- 2. Start Python 3.9 kernel. -->
<!-- 3. Run file. -->

## Changing Variables

<!-- list variables; location in code/what they represent -->



# rounded-Conditional_GAN.ipynb

## Purpose

<!-- short description of what program does -->

<!-- input/output pictures -->

## Steps

<!-- 1. Describe setup -->
<!-- 2. Start Python 3.9 kernel. -->
<!-- 3. Run file. -->

## Changing Variables

<!-- list variables; location in code/what they represent -->



# simple-Conditional_GAN.ipynb

## Purpose

<!-- short description of what program does -->

<!-- input/output pictures -->

## Steps

<!-- 1. Describe setup -->
<!-- 2. Start Python 3.9 kernel. -->
<!-- 3. Run file. -->

## Changing Variables

<!-- list variables; location in code/what they represent -->



# simple-WGAN-GP-V2.ipynb

## Purpose

<!-- short description of what program does -->

<!-- input/output pictures -->

## Steps

<!-- 1. Describe setup -->
<!-- 2. Start Python 3.9 kernel. -->
<!-- 3. Run file. -->

## Changing Variables

<!-- list variables; location in code/what they represent -->



# simple-WGAN-GP.ipynb

## Purpose

<!-- short description of what program does -->

<!-- input/output pictures -->

## Steps

<!-- 1. Describe setup -->
<!-- 2. Start Python 3.9 kernel. -->
<!-- 3. Run file. -->

## Changing Variables

<!-- list variables; location in code/what they represent -->



# simple-WGAN.ipynb

## Purpose

<!-- short description of what program does -->

<!-- input/output pictures -->

## Steps

<!-- 1. Describe setup -->
<!-- 2. Start Python 3.9 kernel. -->
<!-- 3. Run file. -->

## Changing Variables

<!-- list variables; location in code/what they represent -->



# simple-DCGAN.ipynb

## Purpose

<!-- short description of what program does -->

<!-- input/output pictures -->

## Steps

<!-- 1. Describe setup -->
<!-- 2. Start Python 3.9 kernel. -->
<!-- 3. Run file. -->

## Changing Variables

<!-- list variables; location in code/what they represent -->



# simple-GAN.ipynb

## Purpose

First implemented GAN. Trains on pre-generated path data and attempts to output a map with a newly generated path overlayed.

<!-- input/output pictures -->

## Steps

1. Put the map file you want to use in the 'env/' folder, update ```MAP_NAME```, ```MAP_DIMS``` variables accordingly (see next section).
2. Update sweep parameters as desired (see next section).
3. Start Python 3.9 kernel.
4. Run file.

## Changing Variables

Constants (cell 3):
- ```MAP_NAME```: Map file name (without extension).
- ```MAP_DIMS```: Dimensions of map (note that these can be found on the first line of the map file).

Sweep parameters (cell 3):
- ```gen_nodes```: Number of nodes in the Generator.
- ```disc_nodes```: Number of nodes in the Discrminator.
- ```lr```: Learning rate for both Generator and Discriminator.
- ```batch_size```: Batch size used during training.
- ```num_epochs```: Number of epochs used during training.



# sweeper.ipynb

## Purpose

Provides code that can be implemented into GAN programs for hyperparameter sweeping.

<!-- input/output pictures -->

## Steps

1. Put the map file you want to use in the 'env/' folder, update ```MAP_NAME```, ```MAP_DIMS``` variables accordingly (see next section).
2. Create directory to save TensorBoard data in, update ```DATA_DIR``` variable accordingly (see next section).
3. Update sweep parameters as desired (see next section).
4. Start Python 3.9 kernel.
5. Run file.
6. Use TensorBoard to analyze hyperparameter sweep.
    - ```$ tensorboard --logdir [path to DATA_DIR]``` in terminal

## Changing Variables

General constants (cell 2):
- ```MAP_NAME```: Map file name (without extension).
- ```MAP_DIMS```: Dimensions of map (note that these can be found on the first line of the map file).
- ```MAX_DATA_POINTS```: Maximum number of times data will be updated on TensorBoard throughout training.
- ```DATA_DIR```: Name of directory that TensorBoard data will be saved in.

Sweep parameters (cell 3):
- ```features_gen```: Feature number for the Generator.
- ```features_disc```: Feature number for the Discrminator.
- ```noise_channels```: Number of noise channels for both Generator and Discriminator.
- ```lr_gen```: Learning rate for the Generator.
- ```lr_disc```: Learning rate for the Discriminator.
- ```batch_size```: Batch size used during training.
- ```num_epochs```: Number of epochs used during training.
