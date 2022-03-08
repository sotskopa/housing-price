* `create_model.py` uses a .csv file with certain parameters as input and processes training depending on the type of housing. One model is created for each supported housing type.
* `api.py` provides an endpoint to the model used for apartments.
* The `models` directory contains saved models which were trained on preexisting data.

## Predictive model user guide

Using the model requires 1) training and model creation using `create_model.py`, and 2) accessing using `api.py`.

## Required modules

A few modules are required to operate the predictive model. The required models are described below and should be imported before attempting to use the model. They can be easily installed using pip.

* `Tensorflow`
* `Numpy`
* `Pandas`

## Training a model

To train new predictive models, a .csv file containing data to test and train the models on must be supplied. The process of how and where to do this is detailed in the script. For properties where an area is missing, the municipality is set as the area for the property. The script also removes and modifies certain attributes of data in the dataset currently used to train models. If a new dataset with different attributes is to be used, the methods that modify or remove attributes must be changed accordingly. The hyperparameters that determine how the models are trained, such as the activation function, layers, and the number of epochs can also be modified should it be needed. The details of where to change the hyperparameters are also detailed in the script.

When an epoch is finished while training the model a number is displayed which is determined by the loss function. Currently the Mean Absolute Error is used as a loss function. The final number displayed is the accuracy for the finished model.

Saving is done at the end of the training cycle, in a format called `Tensorflow SavedModel`. To load the model elsewhere, you can call `tf.keras.models.load_model(‘models/TYPE’)`, where the argument is replaced with the location of the model. Getting predictions is done by 1) formatting the input data into dictionaries with `Numpy` arrays containing the values, 2) passing this input to the model, and 3) unwrapping the result. These steps are shown in `api.py`.
