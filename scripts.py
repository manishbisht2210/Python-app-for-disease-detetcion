import os
import numpy as np

import skimage
import skimage.data

import matplotlib.pyplot as plt

from skimage import transform
from skimage.color import rgb2gray

import tensorflow as tf 

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".jfif")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(d)
    return images, labels
                
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
train_data_directory = '{}/Diseases/Training'.format(ROOT_PATH)
test_data_directory = '{}/Diseases/Testing'.format(ROOT_PATH)

def process_data():
    
    images, labels = load_data(train_data_directory)

    images = np.array(images)
    # Print the `images` dimensions
    print(images.ndim)
    # Print the number of `images`'s elements
    print(images.size)
    # Print the first instance of `images`
    images[0]

    # labels = np.array(labels)
    # print(labels.ndim)
    # # Print the number of `labels`'s elements
    # print(labels.size)
    # # Count the number of labels
    # print(len(set(labels)))

    # plt.hist(labels, 3)
    # # Show the plot
    # plt.show()

    # Get the unique labels 
    unique_labels = set(labels)

    # Initialize the figure
    plt.figure(figsize=(15, 15))

    # Set a counter
    i = 1

    # For each unique label,
    for label in unique_labels:
        # You pick the first image for each label
        image = images[labels.index(label)]
        # Define 64 subplots 
        plt.subplot(8, 8, i)
        # Don't include axes   
        plt.axis('off')
        # Add a title to each subplot 
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        # Add 1 to the counter
        i += 1
        # And you plot this first image 
        plt.imshow(image)
        
    # Show the plot
    # plt.show()



    # Rescale the images in the `images` array
    images28 = [transform.resize(image, (28, 28)) for image in images]

    # Convert `images28` to an array
    images28 = np.array(images28)
    # Convert `images28` to grayscale
    images28 = rgb2gray(images28)



    # Initialize placeholders 
    x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
    y = tf.placeholder(dtype = tf.int32, shape = [None])

    # Flatten the input data
    images_flat = tf.contrib.layers.flatten(x)

    # Fully connected layer 
    logits = tf.contrib.layers.fully_connected(images_flat, 26, tf.nn.relu)

    # Define a loss function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                        logits = logits))
    # Define an optimizer 
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # Convert logits to label indexes
    correct_pred = tf.argmax(logits, 1)

    # Define an accuracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", correct_pred)

    tf.set_random_seed(1234)
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    for i in range(201):
            print('EPOCH', i)
            _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
            if i % 10 == 0:
                print("Loss: ", loss)
            print('DONE WITH EPOCH')
                                    
                                    
    ##################

    import random

    # Pick 10 random images
    sample_indexes = random.sample(range(len(images28)), 3)
    sample_images = [images28[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]

    # Run the "correct_pred" operation
    predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                            
    # Print the real and predicted labels
    print(sample_labels)
    print(predicted)

    # Display the predictions and the ground truth visually.
    fig = plt.figure(figsize=(3, 3))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2,1+i)
        plt.axis('off')
        color='green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
                 fontsize=12, color=color)
        plt.imshow(sample_images[i],  cmap="gray")

    # plt.show()

    ##################################

    # Load the test data
    test_images, test_labels = load_data(test_data_directory) 
    print(test_labels)

    # Transform the images to 28 by 28 pixels
    test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

    # Convert to grayscale
    test_images28 = rgb2gray(np.array(test_images28))

    # Run predictions against the full test set.
    predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]
    print("The prediction is: ")
    print(predicted)

    # Calculate correct matches 
    match_count = sum([int(int(y) == y_) for y, y_ in zip(test_labels, predicted)])

    # Calculate the accuracy
    accuracy = (match_count / len(test_labels))* 100

    # Print the accuracy
    print("Accuracy: {:.3f}".format(accuracy))

    return str(predicted.item(0))
