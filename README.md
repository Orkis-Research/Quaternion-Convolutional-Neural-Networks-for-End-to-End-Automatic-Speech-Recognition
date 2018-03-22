Quaternion-Valued Convolutional Neural Networks for End-to-End Automatic Speech Recognition
=====================

This repository contains code which reproduces experiments presented in
the paper [link incoming]

Requirements
------------

Install requirements for experiments with pip:
```
pip install numpy tensorflow-gpu keras theano
```
Depending on your Python installation you might want to use anaconda or other tools.
You can also go to the 'Installation' Section and execute the command for an automatic installation.

Installation
------------
Install all the needed dependencies.
```
python setup.py install
```

Experiments
-----------

1. Get help:

    ```
    python scripts/run.py train --help
    ```

2. Run models:

    ```
    python scripts/run.py train --model {real,quaternion} --sf STARTFILTER --nl NUMBEROFLAYERS
    ```

    Other arguments may be added as well; Refer to run.py train --help for
    
      - Optimizer settings
      - Dropout rate
      - Clipping
	  - Saving prefix
      - ...

Citation
--------

Please cite our work as 

```

```
