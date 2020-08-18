# TensorFlow Development Environment

- This is a TensorFlow Development Environment practice.

- [TensorFlowDevelopmentEnvironment](./TensorFlowDevelopmentEnvironment.pdf)



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
- tensorflow
    - sources ./venv/bin/activate
    - pip install tensorflow==1.12.0
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



