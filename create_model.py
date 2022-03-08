import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing
from timeit import default_timer as timer


# Used to measure the training time.
class TimingCallback(callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


class CreateModel:
    def __init__(self, df: pd.DataFrame, type='apartment'):
        self.types = ['apartment', 'villa', 'terrace', 'holiday', 'plot', 'yard']
        # Apartment = Lägenhet
        # Terrace = Radhus
        # Holiday = Fritidshus
        # Plot = Tomt
        # Yard = Gård
        # Note: 'Övrigt' is currently not used in predictions

        if type not in self.types:
            raise Exception('Model does not support type \'{}\''.format(type))
        else:
            self.type = type

        self.dataframe = df[['area', 'sold_date', 'property_type', 'floor', 'rooms', 'price',
                             'square_meter', 'garden_area', 'operating_cost', 'rent', 'year_of_construction']]
        print('Loaded dataframe:\n', self.dataframe)

    def set_type(self, type='apartment'):
        if type not in self.types:
            raise Exception('Model does not support type \'{}\''.format(type))
        else:
            self.type = type

    def get_types(self):
        return self.types

    # Modifies the dataset based on manual observations.
    def create(self):
        input = self.dataframe.copy()
        if self.type == 'apartment':
            input = input.loc[input['property_type'] == 'Bostadsrättslägenhet']
            input = input.drop(['garden_area', 'operating_cost'], 1)  # Apartments generally lack gardens, and
            # operating_cost had many null values.
        else:
            input = input.drop(['floor', 'rent'], 1)  # If not apartment, then we are not dealing with floor or rent.

        if self.type == 'villa':
            input = input.loc[input['property_type'] == 'Villa']
        elif self.type == 'terrace':
            input = input.loc[input['property_type'] == 'Radhus']
        elif self.type == 'holiday':
            input = input.loc[input['property_type'] == 'Fritidshus']
        elif self.type == 'plot':
            input = input.loc[input['property_type'] == 'Tomt'].drop(['operating_cost',
                                                                      'year_of_construction'], 1)  # Most properties
            # of the plot type lacked an operating cost and a year of construction.
        elif self.type == 'yard':
            input = input.loc[input['property_type'] == 'Gård'].drop(['operating_cost'], 1)  # Most properties of the
            # yard type lacked an operating cost
        elif self.type == 'other':
            input = input.loc[input['property_type'] == 'Övrigt']  # Currently not used for predictions. Too unspecific.

        self.train_model(input)

    def train_model(self, df):
        # Split the date attribute into month and year.
        df['year_sold'] = pd.DatetimeIndex(df['sold_date']).year
        df['month_sold'] = pd.DatetimeIndex(df['sold_date']).month
        df = df.drop(['sold_date', 'property_type'], 1).dropna()

        # Features used to predict the label, currently the price attribute.
        housing_features = df.copy()
        housing_labels = housing_features.pop('price')

        # test_size determines how much of the data is used for testing, currently 20%
        x_train, x_test, y_train, y_test = train_test_split(housing_features, housing_labels, test_size=0.2,
                                                            shuffle=True)
        x_train_dict = {name: np.array(value)
                        for name, value in x_train.items()}
        x_test_dict = {name: np.array(value)
                       for name, value in x_test.items()}

        train_ds = tf.data.Dataset.from_tensor_slices((x_train_dict, y_train))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test_dict, y_test))
        train_batches = train_ds.shuffle(len(housing_labels)).batch(16)
        test_batches = test_ds.batch(16)

        # Preprocessing numeric and categorical inputs.
        inputs = {}
        for name, column in housing_features.items():
            dtype = column.dtype
            if dtype == object:
                dtype = tf.string
            else:
                dtype = tf.float32

            inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

        numeric_inputs = {name: input for name, input in inputs.items()
                          if input.dtype == tf.float32}

        # Numeric inputs are normalized and concatenated
        x = layers.Concatenate()(list(numeric_inputs.values()))
        norm = preprocessing.Normalization()
        norm.adapt(np.array(x_train[numeric_inputs.keys()]))
        all_numeric_inputs = norm(x)

        preprocessed_inputs = [all_numeric_inputs]

        # Categorical inputs are one-hot encoded using a lookup mapping. They are then concatenated.
        for name, input in inputs.items():
            if input.dtype == tf.float32:
                continue

            lookup = preprocessing.StringLookup(vocabulary=np.unique(housing_features[name]))
            one_hot = preprocessing.CategoryEncoding(num_tokens=lookup.vocabulary_size())

            x = lookup(input)
            x = one_hot(x)
            preprocessed_inputs.append(x)

        preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

        # Save the preprocessing.
        housing_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

        def housing_model(preprocessing_head, inputs):
            body = tf.keras.Sequential([
                # Currently including four layers with the given dimensionalities.
                layers.Dense(128, activation='swish'),  # The activation function is determined by the activation
                layers.Dense(64),  # variable, currently swish.
                layers.Dense(32),
                layers.Dense(1)
            ])

            preprocessed_inputs = preprocessing_head(inputs)
            result = body(preprocessed_inputs)
            model = tf.keras.Model(inputs, result)

            # The loss variable determines which loss function to use.
            model.compile(loss=tf.losses.MeanAbsoluteError(),
                          optimizer=tf.optimizers.Adam())
            return model

        housing_model = housing_model(housing_preprocessing, inputs)
        cb = TimingCallback()
        # The epochs variable determines the number of epochs, currently 12.
        housing_model.fit(train_batches, epochs=12, callbacks=[cb])
        print('\nFinished training {} model. Total training time: {} s\n'.format(self.type.upper(),
                                                                                 str(int(sum(cb.logs)))))

        # Saving the trained model and printing the loss function value for the model.
        housing_model.save('models/{}'.format(self.type))
        print('\nEvaluating {} model...\n'.format(self.type.upper()))
        housing_model.evaluate(test_batches)


if __name__ == '__main__':
    ddir = os.path.dirname(__file__)
    csv_file = os.path.join(ddir, 'filename.csv')  # Replace with your .csv filename in the same directory as this file.
    dataframe = pd.read_csv(csv_file, sep=',')
    dataframe.loc[dataframe['area'].isnull(), 'area'] = dataframe['municipality']  # Null areas are replaced with the
    # municipality value.
    model = CreateModel(dataframe)
    model.create()  # Default mode is apartment, it is possible to loop through every type by using model.types().
