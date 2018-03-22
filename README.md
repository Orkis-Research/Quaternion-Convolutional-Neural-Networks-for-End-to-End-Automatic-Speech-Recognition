Deep Quaternionary Convolutional Neural Networks
=====================

This repository contains code which reproduces experiments presented in
the paper ...

Requirements
------------

Install requirements for experiments with pip:
```
pip install numpy tensorflow-gpu keras
```
Depending on your Python installation you might want to use anaconda or other tools.


Installation
------------

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
    python scripts/run.py train -w WORKDIR --model {real,complex,quaternion} --seg{chiheb,parcollet} --sf STARTFILTER --nb NUMBEROFBLOCKSPERSTAGE
    ```

    Other arguments may be added as well; Refer to run.py train --help for
    
      - Optimizer settings
      - Dropout rate
      - Clipping
      - ...

Citation
--------

Please cite our work as 

```

```
