Quaternion-Valued Convolutional Neural Networks for End-to-End Automatic Speech Recognition
=====================

This repository contains the code for a simple implementation of the QCNN presented in
the paper [Quaternion Convolutional Neural Networks for End-to-End Automatic Speech Recognition](https://www.researchgate.net/publication/325578506_Quaternion_Convolutional_Neural_Networks_for_End-to-End_Automatic_Speech_Recognition)

Requirements & Installation
---------------------------

Install all the needed dependencies with:
```
python setup.py install
```

Or manually:
```
pip install numpy tensorflow-gpu keras scikit-learn 
```
Depending on your Python installation you might want to use anaconda or other tools.
Note that this code works on python 2.7.13. However it has not been tested on 3.x.

TIMIT
-----
Since the TIMIT dataset is not free, the working_example.py script does not run an experiment on this corpus. However, models used for the experiments
of our paper can be found in models/interspeech_model.py.



Experiment
-----------

1. Run the working example:

    ```
    python working_example.py --model {QDNN, QCNN, DNN, CNN}
    ```

The experiment is conduced on a spoken dialogues corpus which is a set of automatically transcribed human-human telephone conversations from the customer care service (CCS) of the RATP Paris transportation system. The DECODA corpus is composed of 1,242 telephone conversations, which corresponds to about 74 hours of signal. Each conversation has to be mapped to the right theme (8).  