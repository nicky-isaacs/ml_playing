import tensorflow as tf

import preprocess

from typing import List, Dict
from flags import parse_flags
from simpsons_deepnet_model import graph
import numpy as np

IMAGE_HEIGHT_PIXELS = 300
IMAGE_WIDTH_PIXELS = 200


def file_writer_hook(writer: tf.summary.FileWriter):
    scaffold = tf.train.Scaffold(
        summary_op=tf.summary.merge_all()
    )
    return tf.train.SummarySaverHook(
        save_secs=3,
        summary_writer=writer,
        scaffold=scaffold
    )


def labels_to_dict(labels: List[str]) -> Dict[str, int]:
    output = {}
    for i, a in enumerate(labels):
        output[a] = i
    return output


if __name__ == "__main__":
    flags = parse_flags()

    all_annotations = list(preprocess.files_gen(flags.annotations))
    labels = preprocess.all_labels(all_annotations)
    label_dict = labels_to_dict(labels)


    # DEBUGGING
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)



    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)


    def get():
        def build_annotations_dataset_with_int_label(annotations_path):
            all_annotations = list(preprocess.files_gen(annotations_path))

            # Create a dataset of filenames, bounding boxes, string, and int labels
            return tf.data.Dataset.from_generator(
                lambda: ((path, x1, y1, x2, y2, label_dict[label]) for path, x1, y1, x2, y2, label in all_annotations),
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
            .make_one_shot_iterator() \
            .get_next()

        return img_op, label_op


    train_writer = tf.summary.FileWriter('/tmp/simpsons_model')

    hooks = [logging_hook, file_writer_hook(train_writer)]

    model_fn = graph(
        1,
        0.6,
        len(labels),
        flags.learn_rate,
        IMAGE_HEIGHT_PIXELS,
        IMAGE_WIDTH_PIXELS,
        3,
        hooks
    )


    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir="/tmp/simpsons_model",

    )


    estimator.train(
        input_fn=get,
        steps=flags.max_steps,
        hooks=hooks)
