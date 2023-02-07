# Using nn to classify detected hand gesture


## Dependancies

```sh
pip install mediapipe
pip install tensorflow
pip install cvzone
pip install opencv-python
pip install torch
pip install scikit-learn
```


## Obtaining Training Data

`training_data_collection.py` can be ran to collect coordinates data for the 21 interest points from the hand. These data will then be fed into training machine learning model. 

The only change required to make is the path to csv file to be written to. 


## Train a NN Model

Run the NN_Model.py file by `train_python NN_model.py`. It will start training the model and when its done, the model will be saved in the file `gesture_recognition_model.pickle`. Training loss record is saved in `out/33_2_512_1E-02.csv`.

This is a Multiclass Classification problem, so Cross-Entropy is used as for the loss function. `torch.nn.CrossEntropyLoss()` is used to calculate the cross-entropy loss. It takes in the input of probability of classes and the true label value. The ideal loss is log(number_of_gestures). 

Model parameters include:
Optimiser - SGD
Batch_Size - 512
Neurons - [33 33]
Learning_rate - 0.01

## Test the NN Model in real time
Run `python real_time_detection.py`. A video camera screen will pop up and it should detect your hand gestures. Press q to quit the program. 


## How to scale up number of gestures?


