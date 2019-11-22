# Cifar10 Classification with Autoencoder

Several works  point out the usefulness of autoencoders in multiple computer vision tasks such as anomaly detection, image denoising, or even image classification. The encoded layer of an autoencoderprovides a high level representation of the data in its feature maps which can be useful in a classification task. To test this, a Convolutional Autoencoder (CAE) is trained on the CIFAR10 data set, the decoder is discarded and output of the encoder is used as the input of a Convolutional Neural Network (CNN) for image classification.

## System Description 

- GPU: 1xTesla K80 , 2496 CUDA cores, compute 3.7,  12GB(11.173 GB Usable) GDDR5  VRAM

- CPU: 1xsingle core hyper threaded (1 core, 2 threads) Xeon Processors @2.3Ghz (No Turbo Boost) , 46MB Cache

- RAM: ~12.5 GB Available

- Disk: ~310 GB Available 


## Required Modules

- Keras 2.2.5

- Tensorflow 1.15.0

- Scikit-learn 0.21.3

- NumPy 1.17.4

- Matplotlib 3.1.0

- Json 2.0.9

- Pandas 0.25.3


## Project Layout

### /cifar10_modules

Contains modules developed for the project along with their .ipynb counterparts for easy consultation.

### /history 

Contains training history of autoencoders and classifiers in .json format.

### /models 

Contains trained autoencoder and classifier models in .json format as well as weights of encoder components. The optimal models are saved in this directory under the names 'autoencoder_optimal' and 'classifier_optimal'.

### /plots

Contains plots of training and validation metrics from autoencoders and classifiers.

### autoencoder_config_base.json

Base configuration of the autoencoder. It can be loaded and modified.

### classifier_config_base.json

Base configuration of the classifier resulting. It can be loaded and modified.

### pipeline_optimal.ipynb

IPython notebook with the optimal pipeline: load data, train autoencoder, train classifier, evaluate classifier.

### report.pdf

The technical report for the project. 


## Model Configuration 

### Autoencoder 

The autoencoder configuration is represented as a dictionary with the following keys:

```
autoencoder_config= {'activity_regularizer': False,
 'activity_regularizer_type': 'l1',
 'activity_regularizer_value': 0.001,
 'batch_norm': True,
 'batch_size': 64,
 'callbacks': False,
 'conv_blocks': 3,
 'dropout': True,
 'dropout_value': 0.2,
 'early_stopping': False,
 'early_stopping_delta': 0.1,
 'early_stopping_patience': 10,
 'epochs': 50,
 'gaussian_noise_hidden': False,
 'gaussian_noise_input': False,
 'gaussian_noise_stddev': 0.1,
 'image_shape': [32, 32, 3],
 'init_num_filters': 32,
 'kernel_regularizer': False,
 'kernel_regularizer_type': 'l2',
 'kernel_regularizer_value': 0.001,
 'layers_per_block': 2,
 'loss': 'mean_squared_error',
 'lr': 0.001,
 'optimizer': 'adam'}

```


### Classifier

The classifier configuration is represented as a dictionary with the following keys:

```
classifier_config= {'batch_norm': True,
 'batch_size': 64,
 'callbacks': False,
 'class_weights': False,
 'data_augmentation': True,
 'dense': True,
 'dropout': False,
 'dropout_value': 0.2,
 'early_stopping': False,
 'early_stopping_delta': 0.1,
 'early_stopping_patience': 10,
 'epochs': 100,
 'global_pooling': 'flatten',
 'image_shape': [32, 32, 3],
 'loss': 'categorical_crossentropy',
 'lr': 0.001,
 'optimizer': 'adam',
 'weighted_metrics': None}
```


Not all the available configurations were explored in this project due to temporal constrains, but it is relatively easy to test them by changing configurations with these dictionaries. 

## Instructions

To use this project in Google Colab please follow the instructions below:

1. Place the directory containing the project ('cifar10_classification_autoencoder') inside the 'Colab Notebooks' folder in your Google Drive;

2. Open an ipython notebook inside the project folder and place the following cell on top:

```
import sys
from google.colab import drive
drive.mount('/content/gdrive')
%cd /content/gdrive/My\ Drive/Colab \Notebooks/cifar10_classification_autoencoder
!pwd
sys.path.append('/content/gdrive/My Drive/Colab Notebooks/cifar10_classification_autoencoder/cifar10_modules/')
!pip install matplotlib==3.1.0
```

This sets the project folder as the working directory and adds the '/cifar10_modules' direcory to the system path in order to be able to use modules 'dataset.py' and 'modelling.py'. Additionally, it downgrades Matplotlib version to 3.1.0 as the 3.1.1 version 
was raising issues with Seaborn.

3. Import modules 'dataset.py' and 'modelling.py':

```
from dataset import *
from modelling import *
```

4. Set random seed for the Python random number generator, Numpy, and Tensorflow:

```
set_random_seeds(42)
```

5. To load the normalized ([0, 1]) CIFAR10 dataset split into training, validation (as 22% of the training+validation set) and test sets and with 50% of samples from classes bird, truck and deer removed, use the following intruction: 

```
x_train, x_val, x_test, y_train, y_val, y_test, class_names= load_and_norm(0.22)
```

6. To load the base autoencoder configuration:

```
autoencoder_config= load_config('autoencoder_config_base.json')
```
This is a dictionary, so it is possible to easily change the configuration (e.g., autoencoder_config['epochs']= 100). To save this configuration use the save_config() function with a filename of your choosing as in the instruction below:

```
save_config(autoencoder_config, 'autoencoder_config_base.json')
```

7. To train the autoencoder according to autoencoder_config with training and validation sets:

```
autoencoder= train_autoencoder(x_train, x_val, autoencoder_config, autoencoder_filename, encoder_filename)
```

autoencoder_filename will be used to save the fitted autoencoder model in the '/models' directory and a training and validation loss plot in the '/plots' directory. 

encoder_filename will be used to save the weights of the encoder component in the '/models' directory.  

8. To generate a random list of images and visualize the autoencoder predictions for these images:

```
x_pred= autoencoder.predict(x_val)
image_list= random_images(y_val, class_names)
show_image_list(x_val,y_val,class_names,image_list)
show_image_list(x_pred,y_val,class_names,image_list)
```

9. To load the configuration of the base classifier:

```
classifier_config= load_config('classifier_config_base.json')
```

10. To train the classifier based on a classifier_config:

```
classifier= train_classifier(x_train, x_val, y_train, y_val,
                             autoencoder_config, classifier_config,
                             encoder_filename, classifier_filename)

```

autoencoder_config is used to generate the first part of the classifier. 

encoder_filename is used to load the weights of the trained encoder onto the correposnding layers of the classifier. 

classifier_filename is used to save the trained classifier in the '/models' directory,  loss and accuracy plots in the '/plots' folder and history in the '/history' directory.  

11. To evaluate the classifier and build confusion matrix and classification report:

```
y_pred= classifier_predict_evaluate(x_val, y_val, classifier, class_names)
```

