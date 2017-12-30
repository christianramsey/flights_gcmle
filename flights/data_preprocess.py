from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.metrics as tfmetrics
import tensorflow as tf
import numpy as np




# dataset = tf.contrib.timeseries.CSVReader('DelayedFlights.csv')
# print(dataset)


CSV_COLUMNS = ('Cancelled,DepDelay,TaxiOut,Distance').split(',')
LABEL_COLUMN = 'Cancelled'
DEFAULTS = [[0.0], [0.0], [0.0], [0.0]]



        # read CSV
filename_queue = np.array(['DelayedFlights.csv'])
reader = tf.TextLineReader()
_, value = reader.read_up_to(filename_queue, 100)
value_column = tf.expand_dims(value, -1)
columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)

features = dict(zip(CSV_COLUMNS, columns))
label = features.pop(LABEL_COLUMN)



# print(read_dataset("DelayedFlights.csv"))