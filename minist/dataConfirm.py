from tensorflow.examples.tutorials.mnist import input_data

minist = input_data.read_data_sets("/Users/iig-apple-3/Documents/Data/minist/", one_hot=True)

print "Training data size: ", minist.train.num_examples