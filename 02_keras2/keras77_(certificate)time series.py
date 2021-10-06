import csv
import tensorflow as tf
import numpy as np
import urllib
import pandas as pd

urllib.request.urlretrieve('https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv', 'sunspots.csv')
sunspots = pd.read_csv('sunspots.csv', sep=",")
 
print(sunspots)

# Index(['Unnamed: 0', 'Date', 'Monthly Mean Total Sunspot Number'], dtype='object')
  


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1) 


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')

	# Your data should be loaded into 2 Python lists called time_step
	# and sunspots. They are decleared here.
    time_step = []
    sunspots = []
 
    with open('sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(# YOUR CODE HERE)
        time_step.append(# YOUR CODE HERE)

	# You should use numpy to create 
	# - your series from the list of sunspots
	# - your time details from the list of time steps
    series = # YOUR CODE HERE

# # DO NOT CHANGE THIS CODE
#     window_size = 30
#     batch_size = 32
#     shuffle_buffer_size = 1000





#     time = np.array(time_step)

# 	# You should split the dataset into training and validation splits
# 	# At time 3000. So everything up to 3000 is training, and everything
# 	# after 3000 is validation. Write the code below to achieve that.
#     split_time = 3000
#     time_train = # YOUR CODE HERE
#     x_train = # YOUR CODE HERE
#     time_valid = # YOUR CODE HERE
#     x_valid = # YOUR CODE HERE

#     # DO NOT CHANGE THIS CODE
#     window_size = 30
#     batch_size = 32
#     shuffle_buffer_size = 1000


#     tf.keras.backend.clear_session()
#     # You can use any random seed you want. We use 51. :)
#     tf.random.set_seed(51)
#     np.random.seed(51)
#     train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    


#     model = tf.keras.models.Sequential([
#       # YOUR CODE HERE. DO NOT CHANGE THE FINAL TWO LAYERS FROM BELOW
#       tf.keras.layers.Dense(1),
#       # The data is not normalized, so this lambda layer helps
#       # keep the MAE in line with expectations. Do not modify.
#       tf.keras.layers.Lambda(lambda x: x * 400)
#     ])


#     # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
#     return model


# # Note that you'll need to save your model as a .h5 like this
# # This .h5 will be uploaded to the testing infrastructure
# # and a score will be returned to you
# if __name__ == '__main__':
#     model = solution_model()
#     model.save("mymodel.h5")




