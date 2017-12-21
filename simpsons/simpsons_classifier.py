import tensorflow as tf
import sys
import argparse
from shutil import copyfile, rmtree
import os
from ProgressBar import ProgressBar

IMAGE_HEIGHT_PIXELS = 300
IMAGE_WIDTH_PIXELS = 200
TOTAL_PIXELS = 300 * 200 * 3
TOTAL_CLASSES = 18


class Flags:

    def __init__(self, prediction_output_base_dir: str, max_steps: int, training_data: str, test_data: str, learn_rate: float, logs_dir: str):
        self.prediction_output_base_dir = prediction_output_base_dir
        self.max_steps = max_steps
        self.training_data = training_data
        self.test_data = test_data,
        self.learn_rate=learn_rate
        self.logs_dir=logs_dir

def parse_flags() -> Flags:
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-steps',
                        dest='steps',
                        type=int,
                        default=1,
                        help="number of epochs to train (default 1)")
    
    parser.add_argument('--prediction-dir',
                        dest='prediction_output_base_dir',
                        type=str,
                        default='/tmp/predictions',
                        help="location of the output predictions (default /tmp/predictions)")

    parser.add_argument('--training-data',
                        dest='training_data',
                        type=str,
                        help="location of the training data tfdata file")

    parser.add_argument('--test-data',
                        dest='test_data',
                        type=str,
                        help="location of the training data tfdata file")

    parser.add_argument('--learn-rate',
                        dest='learn_rate',
                        type=float,
                        default=0.001,
                        help="location of the training data tfdata file")

    parser.add_argument('--logs-dir',
                        dest='logs_dir',
                        type=str,
                        default='/tmp/simpsons_logs',
                        help='where to log tensorflow data (default /tmp/simpsons_logs)')


    args = parser.parse_args()

    all_args=[
        args.prediction_output_base_dir,
        args.steps,
        args.training_data,
        args.test_data,
        args.logs_dir,
        args.learn_rate,
    ]

    # Ensure that all the flags are defined
    for arg in all_args:
        if None == arg:
            parser.print_usage()
            parser.error('Missing required arguments')

    return Flags(
        prediction_output_base_dir=args.prediction_output_base_dir,
        max_steps=args.steps,
        training_data=args.training_data,
        test_data=args.test_data,
        learn_rate=args.learn_rate,
        logs_dir=args.logs_dir,
    )

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

    return tf.reshape(parsed_img_tensor, (TOTAL_PIXELS,)), tf.reshape(parsed_one_hot_tensor, (TOTAL_CLASSES,)), parsed_label, parsed_path

def put_in_predicted_location(base_sort_dir, location, from_file):
    location = os.path.join(base_sort_dir, location)
    try:
        os.mkdir(location)
    except FileExistsError:
        pass

    filename = os.path.basename(from_file)
    copyfile(from_file, os.path.join(location, filename))

def variable_summaries(var):
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
    with tf.name_scope('w'):
        variable_summaries(w)
    with tf.name_scope('b'):
        variable_summaries(b)
    tf.summary.scalar('cross entropy', cross_entropy)

if __name__ == "__main__":
    flags=parse_flags()
    try:
        rmtree(flags.prediction_output_base_dir)
        os.mkdir(flags.prediction_output_base_dir)
    except Exception:
        os.mkdir(flags.prediction_output_base_dir)


    dataset = tf.data.TFRecordDataset(flags.training_data)

    data_iter = dataset\
        .map(deserialize_example, num_parallel_calls=8)\
        .repeat()\
        .shuffle(1000)\
        .batch(10)\
        .make_one_shot_iterator()  # type: tf.data.Iterator

    inputs_op, true_values_op, label_op, path = data_iter.get_next()

    # Create the model
    x = tf.placeholder(tf.float32, [None, TOTAL_PIXELS])
    W: tf.Variable = tf.Variable(tf.zeros([TOTAL_PIXELS, TOTAL_CLASSES]))
    b: tf.Variable = tf.Variable(tf.zeros([TOTAL_CLASSES]))

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
    merged = tf.summary.merge_all()
    sess = tf.InteractiveSession()
    train_writer = tf.summary.FileWriter(flags.logs_dir + '/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(flags.logs_dir + '/test')

    tf.global_variables_initializer().run()

    bar=ProgressBar(0)
    for i in range(flags.max_steps):
        try:
            inputs, true_values, label = sess.run([inputs_op, true_values_op, label_op])
            summary, _ = sess.run([merged,train_step], feed_dict={x: inputs, y_: true_values})
            if (0== i%10):
                train_writer.add_summary(summary, i)
            bar.incr()
            bar.display()
        except tf.errors.OutOfRangeError:
            break

    dataset_test = tf.data.TFRecordDataset(flags.test_data)

    data_iter_test = dataset_test\
        .map(deserialize_example, num_parallel_calls=8)\
        .batch(1)\
        .make_one_shot_iterator()  # type: tf.data.Iterator

    inputs_op, true_values_op, label_op, path_op = data_iter_test.get_next()

    # Test trained model
    while True:
        try:
            softmax_op = tf.nn.softmax(y, 1)
            predicted_position = tf.argmax(y, 1)
            correct_prediction = tf.equal(predicted_position, tf.argmax(y_, 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            inputs, true_values, label, path = sess.run([inputs_op, true_values_op, label_op, path_op])
            accuracy, our_prediction, predicted_position, path = sess.run([accuracy_op, softmax_op, predicted_position, path_op], feed_dict={x: inputs, y_: true_values})
            put_in_predicted_location(flags.prediction_output_base_dir, str(predicted_position[0]), path[0].decode('utf-8'))
            print("\nTest accuracy: %f" % accuracy)
        except tf.errors.OutOfRangeError:
            break