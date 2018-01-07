import tensorflow as tf

import preprocess

from typing import List, Dict
from flags import parse_flags
from simpsons_deepnet_model import graph
import os
from shutil import copyfile
import random

IMAGE_HEIGHT_PIXELS = 300
IMAGE_WIDTH_PIXELS = 200

model_dir = '/tmp/simpsons_model'


def put_in_predicted_location(base_sort_dir, location, from_file):
    location = os.path.join(base_sort_dir, location)
    try:
        os.mkdir(location)
    except FileExistsError:
        pass

    filename = os.path.basename(from_file)
    copyfile(from_file, os.path.join(location, filename))


def labels_to_dict(labels: List[str]) -> Dict[str, int]:
    output = {}
    for i, a in enumerate(labels):
        output[a] = i
    return output


if __name__ == "__main__":
    flags = parse_flags()

    all_annotations = list(preprocess.files_gen(flags.annotations))
    labels = preprocess.all_labels(all_annotations)
    known_labels = labels_to_dict(labels)
    unknown_label = len(known_labels) + 1

    # DEBUGGING
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)



    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)


    def get_train_data():
        def build_annotations_dataset_with_int_label(annotations_path):
            all_annotations = list(preprocess.files_gen(annotations_path))

            # Do not bias towards samples which are first in the file, as it is alphabetical
            random.shuffle(all_annotations)

            # Create a dataset of filenames, bounding boxes, string, and int labels
            return tf.data.Dataset.from_generator(
                lambda: ((path, x1, y1, x2, y2, known_labels[label]) for path, x1, y1, x2, y2, label in
                         all_annotations),
                (tf.string,
                 tf.int32,
                 tf.int32,
                 tf.int32,
                 tf.int32,
                 tf.int32)
            )  # type: tf.data.Dataset

        def read_crop_resize(path, x1, y1, x2, y2, label_op):
            image_data = tf.read_file(path)
            decoded = tf.image.decode_jpeg(image_data)
            as_float = tf.image.convert_image_dtype(decoded, tf.float32, saturate=True)
            resized = tf.image.resize_bilinear(tf.expand_dims(as_float, axis=0), (300, 200))
            return resized, label_op

        img_op, label_op = build_annotations_dataset_with_int_label(flags.annotations) \
            .map(read_crop_resize, num_parallel_calls=8) \
            .batch(10) \
            .repeat() \
            .make_one_shot_iterator() \
            .get_next()

        return img_op, label_op


    def get_eval_data():
        def build_annotations_dataset_with_int_label(test_set_path):
            def label():
                for path, l in preprocess.parse_test_set(test_set_path):
                    byte_labl = str.encode(l)
                    if byte_labl in known_labels:
                        yield (path.encode("utf-8"), known_labels[byte_labl])
                    else:
                        yield (path.encode("utf-8"), unknown_label)

            # Create a dataset of filenames, bounding boxes, string, and int labels
            return tf.data.Dataset.from_generator(
                label,
                (tf.string, tf.int32)
            )  # type: tf.data.Dataset

        def read_crop_resize(path, label):
            image_data = tf.read_file(path)
            decoded = tf.image.decode_jpeg(image_data)
            as_float = tf.image.convert_image_dtype(decoded, tf.float32, saturate=True)
            resized = tf.image.resize_bilinear(tf.expand_dims(as_float, axis=0), (300, 200))
            return resized, label

        img_op, label_op = build_annotations_dataset_with_int_label(flags.test_set) \
            .map(read_crop_resize, num_parallel_calls=8) \
            .batch(5) \
            .make_one_shot_iterator() \
            .get_next()

        return img_op, label_op


    hooks = [logging_hook]

    device = None
    if flags.enable_gpu:
        device = '/gpu:0'
    else:
        device = '/cpu:0'


    model_fn = graph(
        3,
        0.6,
        len(labels),
        flags.learn_rate,
        IMAGE_HEIGHT_PIXELS,
        IMAGE_WIDTH_PIXELS,
        3,
        hooks,
        model_dir,
        flags.summary_interval,
        device
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=tf.estimator.RunConfig(
            model_dir=model_dir,
            save_summary_steps=1
        )
    )

    tf.logging.set_verbosity(tf.logging.DEBUG)

    if flags.mode == tf.estimator.ModeKeys.TRAIN:
        estimator.train(
            input_fn=get_train_data,
            steps=flags.max_steps,
            hooks=hooks)

    elif flags.mode == tf.estimator.ModeKeys.EVAL:
        estimator.evaluate(
            input_fn=get_eval_data,
            hooks=hooks)
    else:
        print('Only train and eval are supported')
