import tensorflow as tf

import preprocess

from typing import List, Dict
from flags import parse_flags
from simpsons_deepnet_model import graph
import random

IMAGE_HEIGHT_PIXELS = 300
IMAGE_WIDTH_PIXELS = 200

model_dir = '/tmp/simpsons_model'

def file_writer_hook(writer: tf.summary.FileWriter):
    scaffold = tf.train.Scaffold(
        summary_op=tf.summary.merge_all()
    )
    return tf.train.SummarySaverHook(
        save_secs=3,
        scaffold=scaffold,
        output_dir=model_dir,
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

            # Do not bias towards samples which are first in the file, as it is alphabetical
            random.shuffle(all_annotations)

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
            .batch(10) \
            .repeat() \
            .make_one_shot_iterator() \
            .get_next()

        return img_op, label_op



    hooks = [logging_hook]

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
        flags.summary_interval
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
    estimator.train(
        input_fn=get,
        steps=flags.max_steps,
        hooks=hooks)
