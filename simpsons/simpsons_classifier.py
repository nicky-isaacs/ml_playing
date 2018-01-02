import os
import numpy as np
from shutil import copyfile, rmtree

import tensorflow as tf

from ProgressBar import ProgressBar
from flags import parse_flags, Flags

IMAGE_HEIGHT_PIXELS = 300
IMAGE_WIDTH_PIXELS = 200
TOTAL_PIXELS = 300 * 200 * 3
TOTAL_CLASSES = 18


def deserialize_example(example_bytes):
    parsed = tf.parse_single_example(
        example_bytes,
        features={
            'image': tf.FixedLenFeature([], dtype=tf.string),
            'one_hot': tf.FixedLenFeature([], dtype=tf.string),
            'label': tf.FixedLenFeature([], dtype=tf.string),
            'image_path': tf.FixedLenFeature([], dtype=tf.string),
        },
        name='parse_example'
    )

    parsed_img_tensor = tf.parse_tensor(parsed['image'], tf.float32, name='parse_image')
    parsed_one_hot_tensor = tf.parse_tensor(parsed['one_hot'], tf.float32, name='parse_one_hot')
    parsed_label = parsed['label']
    parsed_path = parsed['image_path']

    return tf.reshape(parsed_img_tensor, (TOTAL_PIXELS,)), tf.reshape(parsed_one_hot_tensor,
                                                                      (TOTAL_CLASSES,)), parsed_label, parsed_path


def put_in_predicted_location(base_sort_dir, location, from_file):
    location = os.path.join(base_sort_dir, location)
    try:
        os.mkdir(location)
    except FileExistsError:
        pass

    filename = os.path.basename(from_file)
    copyfile(from_file, os.path.join(location, filename))


def variable_summaries(var: tf.Variable):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def setup_tensorboard(w, b, cross_entropy):
    """Given the variables and tensors we care about, attach scoped metrics"""
    with tf.name_scope('w'):
        variable_summaries(w)
    with tf.name_scope('b'):
        variable_summaries(b)
    tf.summary.scalar('cross entropy', cross_entropy)


def create_test_dataset(flags: Flags):
    dataset_test = tf.data.TFRecordDataset(flags.test_data)
    return dataset_test \
        .repeat() \
        .map(deserialize_example, num_parallel_calls=8) \
        .batch(1) \
        .make_one_shot_iterator()  # type: tf.data.Iterator


def build_test_dataset(flags: Flags) -> tf.data.Dataset:
    dataset_test = tf.data.TFRecordDataset(flags.test_data)
    return dataset_test \
        .map(deserialize_example, num_parallel_calls=8) \
        .batch(1)


if __name__ == "__main__":
    flags = parse_flags()
    try:
        rmtree(flags.prediction_output_base_dir)
        os.mkdir(flags.prediction_output_base_dir)
    except Exception:
        os.mkdir(flags.prediction_output_base_dir)

    dataset = tf.data.TFRecordDataset(flags.training_data)

    data_iter = dataset \
        .map(deserialize_example, num_parallel_calls=8) \
        .repeat() \
        .shuffle(10000) \
        .batch(10) \
        .make_one_shot_iterator()  # type: tf.data.Iterator

    inputs_op, true_values_op, label_op, path = data_iter.get_next()

    # Create the model
    x = tf.placeholder(tf.float32, [None, TOTAL_PIXELS])
    W = tf.Variable(tf.zeros([TOTAL_PIXELS, TOTAL_CLASSES]))  # type: tf.Variable
    b = tf.Variable(tf.zeros([TOTAL_CLASSES]))  # type: tf.Variable

    # y = ax+b, learn a and b
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, TOTAL_CLASSES])

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
    train_step = tf.train.GradientDescentOptimizer(flags.learn_rate).minimize(cross_entropy)

    setup_tensorboard(W, b, cross_entropy)

    sess = tf.InteractiveSession()
    train_writer = tf.summary.FileWriter(flags.logs_dir + '/train',
                                         sess.graph)

    train_acc_report_var = tf.Variable(initial_value=0.0, dtype=tf.float32)
    test_acc_report_var = tf.Variable(initial_value=0.0, dtype=tf.float32)

    with tf.name_scope('train_accuracy'):
        variable_summaries(train_acc_report_var)

    with tf.name_scope('test_accuracy'):
        variable_summaries(test_acc_report_var)

    acc_report_test_data = build_test_dataset(flags).batch(10).repeat().make_one_shot_iterator()
    acc_inputs_op, acc_true_values_op, _, _ = acc_report_test_data.get_next()


    def update_train_acc():
        _predicted_position = tf.argmax(y, 1)
        _correct_prediction = tf.equal(_predicted_position, tf.argmax(y_, 1))
        acc_op = tf.reduce_mean(tf.reduce_mean(tf.cast(_correct_prediction, tf.float32)))
        update_acc_op = train_acc_report_var.assign(
            (acc_op + tf.convert_to_tensor(train_acc_report_var)) / tf.constant(2.0, tf.float32))
        _inputs, _true_values = sess.run([acc_inputs_op, acc_true_values_op])
        sess.run(update_acc_op, feed_dict={x: _inputs, y_: _true_values})


    def update_test_acc(_inputs, _true_values):
        _predicted_position = tf.argmax(y, 1)
        _correct_prediction = tf.equal(_predicted_position, tf.argmax(y_, 1))
        acc_op = tf.reduce_mean(tf.reduce_mean(tf.cast(_correct_prediction, tf.float32)))
        update_acc_op = test_acc_report_var.assign(
            (acc_op + tf.convert_to_tensor(train_acc_report_var)) / tf.constant(2.0, tf.float32))
        new_acc = sess.run(update_acc_op, feed_dict={x: _inputs, y_: _true_values})
        print("\nTest accuracy: %f" % new_acc)


    tf.global_variables_initializer().run()

    bar = ProgressBar(flags.max_steps)
    merged = tf.summary.merge_all()
    for i in range(flags.max_steps):
        try:
            inputs, true_values, label = sess.run([inputs_op, true_values_op, label_op])
        except tf.errors.OutOfRangeError:
            break
        summary, _ = sess.run([merged, train_step], feed_dict={x: inputs, y_: true_values})
        if i % flags.tf_summary_interval is 0:
            update_train_acc()
            update_test_acc(inputs, true_values)
            train_writer.add_summary(summary, i)
        bar.incr()
        bar.display()

    data_iter_test = build_test_dataset(flags).make_one_shot_iterator()

    inputs_op, true_values_op, label_op, path_op = data_iter_test.get_next()

    # Test trained model
    acc_sum = 0.0
    while True:
        try:
            predicted_position = tf.argmax(y, 1)
            correct_prediction = tf.equal(predicted_position, tf.argmax(y_, 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            inputs, true_values, label, path = sess.run([inputs_op, true_values_op, label_op, path_op])
            accuracy, predicted_position, path = sess.run([accuracy_op, predicted_position, path_op],
                                                          feed_dict={x: inputs, y_: true_values})
            put_in_predicted_location(flags.prediction_output_base_dir, str(predicted_position[0]),
                                      path[0].decode('utf-8'))
            acc_sum = (acc_sum + accuracy) / 2.0
        except tf.errors.OutOfRangeError:
            break
    print("\nTest accuracy: %f" % acc_sum)
