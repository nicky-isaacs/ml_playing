import numpy as np
import tensorflow as tf
import csv
from PIL import Image
import sys
import os
import traceback

IMAGE_HEIGHT_PIXELS = 300
IMAGE_WIDTH_PIXELS = 200
TOTAL_PIXELS = 300 * 200 * 3


class BoundingBox:
    def __init__(self, upper_left_corner, lower_right_corner):
        self.upper_left_corner = upper_left_corner
        self.lower_right_corner = lower_right_corner


class Label:
    def __init__(self, name, index, total_label_count):
        self.name = name
        self.index = index
        self.total_label_count = total_label_count


class CroppedAndResizedSimpsonsImage:
    def __init__(self, image, label):
        self.image = image
        self.label = label

    # Creates a tensor which is a vector of all the flattened RGB values in the image
    def to_tensor(self):
        # return a tensor of the image in shape [200*300*3]
        image_data = []
        for pixel in self.image.getdata():
            r, g, b = pixel
            image_data += [r, g, b]
        return tf.constant(image_data, tf.float64)


class SimpsonsCSVEntry:
    def __init__(self, absolute_file_path, bounding_box, label):
        self.absolute_file_path = absolute_file_path
        self.bounding_box = bounding_box
        self.label = label


def crop_and_resize(img, bounding_box):
    return img \
        .crop((bounding_box.upper_left_corner[0], bounding_box.upper_left_corner[1], bounding_box.lower_right_corner[0],
               bounding_box.lower_right_corner[1])) \
        .resize((IMAGE_HEIGHT_PIXELS, IMAGE_WIDTH_PIXELS), Image.ANTIALIAS)


def simpons_csv_to_cropped_or_none(simpsons_csv):
    try:
        image = Image.open(simpsons_csv.absolute_file_path)
        image.verify()
        image = Image.open(simpsons_csv.absolute_file_path)
        if simpsons_csv.bounding_box is not None:
            return CroppedAndResizedSimpsonsImage(
                crop_and_resize(image, simpsons_csv.bounding_box),
                simpsons_csv.label)
        else:
            return None
    except Exception as e:
        # print "Error:" + str(e)
        traceback.print_exc()
        return None


# Format: filepath,x1,y1,x2,y2,character
def read_csv(path):
    with open(path, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            relative_path = row[0]
            csv_base_path = os.path.dirname(path)
            image_path = os.path.abspath(os.path.join(csv_base_path, relative_path))
            yield SimpsonsCSVEntry(image_path, BoundingBox((int(row[1]), int(row[2])), (int(row[3]), int(row[4]))),
                                   row[5])


def sort_and_uniq_labels(names):
    as_set = list(set(names))
    as_set.sort()
    return as_set


if __name__ == "__main__":
    csv_entries = []
    label_strings = []
    for csv_entry in read_csv(sys.argv[1]):
        entry = simpons_csv_to_cropped_or_none(csv_entry)
        if entry is not None:
            csv_entries += [entry]
            label_strings += [entry.label]
        else:
            print(csv_entry)
    labels = sort_and_uniq_labels(label_strings)
    total_labels = len(labels)
    for e in csv_entries:
        label_index = labels.index(e.label)
        vec = np.zeros(total_labels)
        vec[label_index] = 1
        e.one_hot = vec
        print(vec)
    # Create the model
    x = tf.placeholder(tf.float32, [None, TOTAL_PIXELS])
    W = tf.Variable(tf.zeros([TOTAL_PIXELS, total_labels]))
    b = tf.Variable(tf.zeros([total_labels]))
    # y = ax+b, learn a and b
    y = tf.matmul(x, W) + b
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    inputs = []
    true_values = []
    for img in csv_entries:
        inputs += [img.to_tensor()]
        true_values += [img.one_hot]
    for _ in range(1000):
        sess.run(train_step, feed_dict={x: inputs, y_: true_values})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: inputs,
                                        y_: true_values}))

    # check the flag for location of the .csv file
    # read in the csv file, parsed into SimpsonsCSVEntry objects
    # filter out the bad ones
    # apply bounding box to the images
    # resize to AxB where A and B are constants
    # turn these things into tensors
