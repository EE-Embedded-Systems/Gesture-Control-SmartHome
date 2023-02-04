# Using nn to classify detected hand gesture


## Dependancies

```sh
pip install mediapipe
pip install tensorflow
pip install cvzone
pip install opencv-python
```


## Obtaining Training Data

`training_data_collection.py` can be ran to collect coordinates data for the 21 interest points from the hand. These data will then be fed into training machine learning model. 

The only change required to make is the path to csv file to be written to. 