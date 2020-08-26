# TensorFlow Development Environment

- This is a TensorFlow Development Environment practice.

- [TensorFlowDevelopmentEnvironment](Data/TensorFlowDevelopmentEnvironment.pdf)



## Install TensorFlow
- Hardware
    - CPU
    - GPU
    - Cloud TPU
    - Android
    - IOS
    - Embedded Devices
- Operation System
    - Ubuntu 
    - Windows 
    - macOS 
    - Raspbian
- brew
    - https://brew.sh/
    - brew update
    - brew install python@2
- pip
    - tensorflow —Current release for CPU-only (recommended for beginners)
    - tensorflow-gpu —Current release with GPU support (Ubuntu and Windows)
    - tf-nightly —Nightly build for CPU-only (unstable)
    - tf-nightly-gpu —Nightly build with GPU support (unstable, Ubuntu and Windows)
    - curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    - python get-pip.py
- virtualenv
    - sudo pip install -U virtualenv
- venv
    - virtualenv --system-site-packages -p python2.7 ./venv
    - virtualenv --system-site-packages -p python3.7 ./py3
    
- tensorflow
    - source ./venv/bin/activate  # py version = 2.7
    - source ./py3/bin/activate  # py version = 3.7
    
    - pip install tensorflow==1.12.0  # py version = 2.7
    - pip install tensorflow==1.13.1  # py version = 3.7
    
    - pip install jupyter
    - pip install seaborn
    - pip install matplotlib
    - pip install pandas
    - pip install numpy
    - ...
    - pip list
    - python -c "import tensorflow as tf"
    - deactivate


## CMD Hello TensorFlow

- code:

```python
import tensorflow as tf
hello = tf.constant("hello Tensorflow")
sess = tf.Session()
sess.run(hello)
print(sess.run(hello))

# warning
# I tensorflow/core/platform/cpu_feature_guard.cc:141] 
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
```

- AVX2 Instruction Set CPUS
    - Intel
        - https://en.wikipedia.org/wiki/Intel
    - AMD
        - https://en.wikipedia.org/wiki/Advanced_Micro_Devices


## Jupyter Notebook TensorFlow

- Install Jupyter: 
```shell script
source ./venv/bin/activate

pip install jupyter

python –m ipykernel install --user --name=venv  # py version = 2.7
python3.7 –m ipykernel install --user --name=py3  # py version = 3.7

# check kernel list
jupyter kernelspec list

# open jupyter
jupyter notebook 

deactivate
```


- Demo
	- Neural Network Overview
	- MNIST Dataset Overview
	- link: [MNISTDatasetNeuralNetwork](Data/MNISTDatasetNeuralNetwork.ipynb)





## Docker TensorFlow

- Install Docker App
- Run Docker App
- Pull a TensorFlow Docker image
    - docker pull tensorflow/tensorflow:nightly-jupyter
- Start a TensorFlow Docker container
    - docker run -it -p 8888:8888 -v $PWD:/Users/rmliu/Desktop/tensorflow:nightly-jupyter
    - eg：(notebook-examples-path):/Users/rmliu/Desktop/tensorflow:nightly-jupyter










