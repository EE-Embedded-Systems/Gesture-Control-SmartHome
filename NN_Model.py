import torch
from torch import nn
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold

###
# Training Device
###

if torch.cuda.is_available():
    print("GPU Detected")
    DEVICE = torch.device("cuda:0")
    # WARNING: Uncomment when training final model
    # DEVICE = torch.device("cpu")
else:
    print("GPU Not Detected")
    DEVICE = torch.device("cpu")

###
# PyTorch Model
###


class NeuralNetwork(nn.Module):
    def __init__(self, neurons, inputs=63, outputs=3):
        """Initializes a PyTorch multi-layer linear nueral network.

        Args:
            inputs (int): Number of features in the input. Defaults to 13.
            outputs (int): Number of features in the output. Defaults to 1.
            neurons (list): Number of Neurons in each hidden layer. Length of the list corresponds to the number of layers.
        """
        super(NeuralNetwork, self).__init__()

        # Initialize NN Network
        input_dimension = inputs
        layers = []
        for output_dimension in neurons:
            layers.append(nn.Linear(input_dimension, output_dimension))
            layers.append(nn.ReLU())
            input_dimension = output_dimension
        layers.append(nn.Linear(input_dimension, outputs))
        layers.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*layers).to(DEVICE)

    def forward(self, x):
        """Performs forward pass through the network.

        Args:
            x (np.ndarray): Input array of shape (batch_size, input_dimension).
        """
        return self.layers(x)

###
# Early Stopping
###


class EarlyStopper:
    def __init__(self, patience=3, error=0.0001):
        """Initializes early stopper Class.

        Args:
            patience (int): Number of consecutive times the validation loss hasn't changed. Defaults to 3.
            error (int): Loss tolerance to to indicate whether the validation loss has changed. Defaults to 0.

        Note: Class is inspired by https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
        """
        self.patience = patience
        self.counter = 0
        self.last_validation_loss = np.inf
        self.stop = False
        self.error = error

    def __call__(self, validation_loss):
        """Updates counter and minimum validation loss of early stopper.

        Args:
            validation_loss (float): Current validation loss.
        """

        if validation_loss < self.last_validation_loss:
            self.counter = 0
            self.last_validation_loss = validation_loss
        elif self.counter >= self.patience:
            self.stop = True
        else:
            self.counter += 1

###
# Loss Logger
###


class LossLogger:
    """Logs Training Loss and Validation Loss at Each Epoch."""

    def __init__(self):
        self.logs = []

    def __call__(self, epoch, training_loss, validation_loss):
        """Logs Trainings and Validation Loss.

        Args:
            epoch (int): Current epoch.
            training_loss (float): Training loss of the current epoch.
            validation_loss (float): Validation loss of the current epoch.
        """
        self.logs.append(
            {
                "epoch": epoch,
                "Training Loss": training_loss,
                "Validation Loss": validation_loss,
            }
        )

    def save(self, name, folder="out"):
        """Saves the losses into a Pandas DataFrame.

        Args:
            name (str): File name.
            folder (str): Name of folder to save CSV into. Defaults to "out".
        """
        logs = pd.DataFrame(self.logs)
        logs.to_csv(f"{folder}/{name}.csv", index=False)


class Classifier:
    def __init__(
        self,
        x,
        nb_epoch=5000,
        batch_size=512,
        learning_rate=0.01,
        neurons=[21, 21, 21],
        optimiser="adam",
    ):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
            - batch_size {int} -- Number of records in a batch.
            - learning rate {float} -- Learning rate.
            - neurons {list} -- List of integers that indicate the number of neurons in each hidden layer.
            - optimiser {string} -- Gradient descent optimiser. Defaults to adam

        """

        # Define Model
        self.model = NeuralNetwork(neurons)

        # Define Hyperparameters
        self.nb_epoch = nb_epoch
        self.optimiser = optimiser
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.neurons = neurons

        self.set_name()

    def set_name(self):
        """Sets Classifier's name based on Hyperparameters"""
        neuron = str(self.neurons[-1])
        layers = str(len(self.neurons))

        self.name = "_".join(
            [neuron, layers, str(self.batch_size),
             "{:.0E}".format(self.learning_rate)]
        )

    def fit(self, x, y, x_validation=None, y_validation=None):
        """
        Classifier training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Classifier} -- Trained model.
        """
        # Convert form df to tensor
        np_x, np_y = x.values, y.values

        # Move over to GPU (if available)
        x_train, y_train = torch.from_numpy(np_x).to(
            DEVICE),  torch.from_numpy(np_y).to(DEVICE)
        data_length = x_train.size()[0]

        # Initialize Loss Function, Optimizer and Stopper.
        criterion = torch.nn.CrossEntropyLoss()
        optimiser = self._create_optimiser(self.optimiser)
        stopper = EarlyStopper(patience=10)

        # Train
        logger = LossLogger()
        for epoch in range(self.nb_epoch):

            permutation = torch.randperm(data_length).to(DEVICE)

            for i in range(0, data_length, self.batch_size):

                # Prepare Data
                idx = permutation[i: i + self.batch_size]
                batch_x, batch_y = x_train[idx].float(), y_train[idx].long()

                # Reset Gradients
                optimiser.zero_grad()

                # Forward Pass
                y_predicted = self.model.forward(batch_x)
                # print(y_predicted)
                # print(batch_y)

                # Loss
                batch_y = batch_y.squeeze()
                loss = criterion(y_predicted, batch_y)

                # Backward Pass
                loss.backward()

                # Update
                optimiser.step()

            # Compute Epoch Loss after training with all batches
            epoch_loss = self.score(x, y)
            if (x_validation is not None) and (y_validation is not None):

                # Compute Validation Loss
                validation_loss = self.score(x_validation, y_validation)

                # Log
                logger(epoch + 1, epoch_loss.item(),
                       validation_loss.item())
                print(
                    "Epoch: %d | Loss (Cross-Entropy): %.4f | Validation Loss (Cross-Entropy): %.4f"
                    % (epoch + 1, epoch_loss.item(), validation_loss.item())
                )

                # Stop
                stopper(validation_loss)
                if stopper.stop:
                    print("Stopping Training!")
                    break

            else:
                # Print Metrics
                logger(epoch + 1, epoch_loss, 0)
                print("Epoch: %d | Loss: %.4f" % (epoch + 1, epoch_loss))

        logger.save(self.name)

        return self

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        x_processed, _ = self._preprocessor(x, training=False)
        return self.model(x_processed.float()).cpu().detach().numpy()

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
        x, y = torch.from_numpy(x.values), torch.from_numpy(y.values)
        loss_fn = nn.CrossEntropyLoss()

        # Calculate Loss
        y_predicted = self.model(x.float())
        return loss_fn(y_predicted, y.long().squeeze())

    def _create_optimiser(self, name):
        """
        Creates Torch Optimiser Object

        Arguments:
            - name {str} -- Name of Optimiser to use

        Returns:
            {torch.optim} -- Pytorch optimiser Object.
        """
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.1)
        elif name == "adadelta":
            return torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError("Optimiser not implemented!")


def split_dataset(data, split=[0.6, 0.2, 0.2]):
    """Splits dataset into training, validation and testing sets.

    Args:
        data (pandas.DataFrame): Pandas DataFrame containing features and labels.
        split (list): List of proportions to split into training, validation and testing datasets.

    Returns:
        train (pandas.DataFrame): Training dataset.
        val (pandas.DataFrame): Validation dataset.
        test (pandas.DataFrame): Testing dataset.
    """

    assert sum(split) == 1, "Splits don't sum to 1!"
    assert len(split) == 3, "Number of splits isn't 3!"

    data_length = len(data)
    train, val, test = np.split(
        data.sample(frac=1),
        [int(split[0] * data_length), int((1 - split[-1]) * data_length)],
    )

    return (train, val, test)


def save_classifier(trained_model):
    """
    Utility function to save the trained classifier model in gesture_recognition_model.pickle
    """
    assert type(trained_model) == Classifier
    # If you alter this, make sure it works in tandem with load_classifier
    with open("gesture_recognition_model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in gesture_recognition_model.pickle\n")


def confusion_matrix(y_predicted, y_gold):
    """ Construct Confusion Matrix

    Args:
        y_predicted (tensor): Probability of each classes (test_size, C)
        y_gold (pd.DataFrame): True Label of classes (test_size,)
    """
    size = y_predicted.shape[1]
    predicted_label = np.argmax(y_predicted.detach().numpy(), axis=1)
    conf_matrix = np.zeros((size, size), dtype=np.int64)
    y_gold = y_gold.values
    for i in range(y_gold.shape[0]):
        true_label = y_gold[i].item()
        pred_label = predicted_label[i]
        conf_matrix[true_label][pred_label] += 1
    return conf_matrix


def train_one_model():
    output_label = "label"

    # Use pandas to read CSV data as it contains various object types
    fist_data = pd.read_csv("Data/fist.csv")
    thumbs_down_data = pd.read_csv("Data/thumbs_down.csv")
    thumbs_up_data = pd.read_csv("Data/thumbs_up.csv")
    combined_data = pd.concat([fist_data, thumbs_down_data, thumbs_up_data])

    # Changing label into values. This is to allow calculation of loss function with cross entropy
    label_map = {'fist': 0, 'thumbs_down': 1, 'thumbs_up': 2}
    combined_data['label'] = combined_data['label'].map(label_map)

    # Split Datasets into Train and Test DataSets
    train, val, test = split_dataset(combined_data)

    # Split Input and Output
    x_train, x_validation, x_test = (
        train.loc[:, combined_data.columns != output_label],
        val.loc[:, combined_data.columns != output_label],
        test.loc[:, combined_data.columns != output_label],
    )
    y_train, y_validation, y_test = (
        train.loc[:, [output_label]],
        val.loc[:, [output_label]],
        test.loc[:, [output_label]],
    )

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    classifier = Classifier(
        x_train,
        batch_size=512,
        learning_rate=0.01,
        optimiser="sgd",
        nb_epoch=3000,
        neurons=[33, 33],
    )
    classifier.fit(x_train, y_train, x_validation, y_validation)
    # classifier = load_classifier()

    print(vars(classifier))
    # Error
    error = classifier.score(x_test, y_test)
    x_test = torch.from_numpy(x_test.values)
    y_predicted = classifier.model(x_test.float())
    conf_mat = confusion_matrix(y_predicted, y_test)
    print("\nClassifier error: {}\n".format(error))
    print("Confusion Matrix: \n", conf_mat)
    save_classifier(classifier)


if __name__ == "__main__":
    train_one_model()
