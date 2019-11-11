# Cifar10 Classification with Autoencoders


# System Description 

The models in this repository were trained in Google Colab with the following specifications:

- GPU: 1xTesla K80 , having 2496 CUDA cores, compute 3.7,  12GB(11.173 GB Usable) GDDR5  VRAM

- CPU: 1xsingle core hyper threaded i.e(1 core, 2 threads) Xeon Processors @2.3Ghz (No Turbo Boost) , 46MB Cache

- RAM: ~12.5 GB Available

- Disk: ~310 GB Available 
 
For every 12hrs or so Disk, RAM, VRAM, CPU cache etc data that is on our alloted virtual machine will get erased 

# Instructions

To use this project in Google Colab please follow the instructions below:

1. Place the directory containing the project ('cifar10_classification_autoencoder') inside the 'Colab Notebooks' folder in your Google Drive;

2. Open an ipython notebook inside the project folder and place the following cel on top:

```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from google.colab import drive
import sys
drive.mount('/content/gdrive')
%cd /content/gdrive/My\ Drive/Colab \Notebooks/cifar10_classification_autoencoder
!pwd
sys.path.append('/content/gdrive/My Drive/Colab Notebooks/cifar10_classification_autoencoder/cifar10_module/')
!pip install matplotlib==3.1.0
```

3. load config

4. change config

5. load data

6. train autoencoder

7. visualize reconstruction

8. train classifier

9. predict and evaluate

10. tune

11. predict and evaluate

