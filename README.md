# Cifar10 Classification with Autoencoder


# System Description 

- GPU: 1xTesla K80 , 2496 CUDA cores, compute 3.7,  12GB(11.173 GB Usable) GDDR5  VRAM

- CPU: 1xsingle core hyper threaded i.e(1 core, 2 threads) Xeon Processors @2.3Ghz (No Turbo Boost) , 46MB Cache

- RAM: ~12.5 GB Available

- Disk: ~310 GB Available 


# Module requirements:

- Keras 2.2.5

- Tensorflow 1.15.0

- Scikit-learn 0.21.3

- NumPy 1.17.4

- Matplotlib 3.1.0

- Json 2.0.9

- Pandas 0.25.3

# Instructions

To use this project in Google Colab please follow the instructions below:

1. Place the directory containing the project ('cifar10_classification_autoencoder') inside the 'Colab Notebooks' folder in your Google Drive;

2. Open an ipython notebook inside the project folder and place the following cell on top:

```
import sys
from google.colab import drive
drive.mount('/content/gdrive')
%cd /content/gdrive/My\ Drive/Colab \Notebooks/cifar10_classification_autoencoder
!pwd
sys.path.append('/content/gdrive/My Drive/Colab Notebooks/cifar10_classification_autoencoder/cifar10_module/')
!pip install matplotlib==3.1.0
```

This sets the project folder as the working directory and adds the 'cifar10_module' direcory to the system path in order to be able to use modules 'dataset.py' and 'modelling.py'. Additionally, it downgrades matplotlib version to 3.1.0 as the 3.1.1 version 
was raising issues with Seaborn.

2. Import modules 'dataset.py' and 'modelling.py':

```
from dataset import *
from modelling import *
```

3. Set random seed for the Python random number generator, Numpy, and Tensorflow:

```
set_random_seeds(42)
```

4. To load the normalized (in [0, 1] CIFAR10 dataset split into training, validation (as 22% of the training+validation set) and test sets and with 50% of samples from classes bird, truck and deer removed, use the following intruction: 

```
x_train, x_val, x_test, y_train, y_val, y_test, class_names= load_and_norm(0.22)
```

4. To load the optimal autoencoder configuration:

```
autoencoder_config= load_config('autoencoder_config_optimal.json')
```
This is a dictionary, so it is possible to easily change the configuration (e.g., autoencoder_config['epochs']=100). To save this configuration use the save_config() function with a filename of your choosing as in the instruction below:

```
save_config(autoencoder_config, 'autoencoder_config_optimal.json')
```

5. To train the autoencoder according to 'autoencoder_configuration' with training and validation data:

```
autoencoder= train_autoencoder(x_train, x_val, autoencoder_config, autoencoder_filename, encoder_filename)
```

autoencoder_filename will be used to save the fitted autoencoder model in the '/models' directory, the training history in the '/history' directory and a training and validation loss plot in the '/plots' folder encoder_filename will be used to save the weights of the encoder component in the '/weights' directory.  

6. train autoencoder

7. visualize reconstruction

8. train classifier

9. predict and evaluate

10. tune

11. predict and evaluate

