# Using Deep Neural Network to classify detected hand gesture


## Dependancies

```sh
pip install -r requirements.txt
```


## Obtaining Training Data

`training_data_collection.py` can be ran to collect coordinates data for the 21 interest points from the hand. These data will then be fed into training machine learning model. 

The only change required to make is the path to csv file to be written to. 

## Usage 

### Train a NN Model

Run the NN_Model.py file by `train_python NN_model.py`. It will start training the model and when its done, the model will be saved in the file `gesture_recognition_model.pickle`. Training loss record is saved in `out/33_2_512_1E-02.csv`.

Notes: This is a Multiclass Classification problem, so Cross-Entropy is used as for the loss function. `torch.nn.CrossEntropyLoss()` is used to calculate the cross-entropy loss. It takes in the input of probability of classes and the true label value. The ideal loss is log(number_of_gestures). 

Ideal Model parameters include:
Optimiser - SGD
Batch_Size - 512
Neurons - [33 33]
Learning_rate - 0.03

### Test the NN Model in real time
Run `python real_time_detection.py`. A video camera screen will pop up and it should detect your hand gestures. Press q to quit the program. 


## Directory Structure

### data
Contains all the training data collected 

### out
Contains all the training log loss history. 

## Final Model Parameters Obtained

Cross-Entropy error: 0.013644796796143055

Confusion Matrix: 
 [[592   1   0   0   0   0   3   1]
 [  1 653   0   0   0   0   0   0]
 [  0   0 626   0   0   0   0   0]
 [  0   0   0 639   0   0   0   0]
 [  0   0   0   0 618   0   0   0]
 [  0   0   0   0   0 595   0   0]
 [  0   0   0   0   0   0 590   0]
 [  0   0   0   0   0   0   0 614]]

Using the given confusion matrix, we can calculate these metrics as follows:

TP = 611, FP = 4, FN = 8, TN = 3859
Accuracy = (611 + 3859) / (611 + 4 + 8 + 3859) = 0.9978 (or 99.78%)
Precision = 611 / (611 + 4) = 0.9935 (or 99.35%)
Recall = 611 / (611 + 8) = 0.9870 (or 98.70%)
F1 Score = 2 * (0.9935 * 0.9870) / (0.9935 + 0.9870) = 0.9903 (or 99.03%)

Therefore, the accuracy is 99.78%, precision is 99.35%, recall is 98.70%, and F1 score is 99.03%.


