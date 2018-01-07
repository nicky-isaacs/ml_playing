import csv
import os
from typing import Iterator, Tuple, Generator, List

import tensorflow as tf
import sys
import random
from PIL import Image

from ProgressBar import ProgressBar


def files_gen(annotations_path: str) -> Generator[Tuple[str, int, int, int, int, str], None, None]:
    with open(annotations_path, 'rt') as csvfile:
        annotation_reader = csv.reader(csvfile, delimiter=',')  # type: Iterator[Tuple[str, str, str, str, str, str]]
        for row in annotation_reader:
            relative_path = row[0]
            csv_base_path = os.path.dirname(annotations_path)
            image_path = os.path.abspath(os.path.join(csv_base_path, relative_path))
            if os.path.exists(image_path):
                yield (image_path.encode('utf-8'),
                       int(row[1]), int(row[2]),  # upper left x,y
                       int(row[3]), int(row[4]),  # lower right x,y
                       row[5].encode('utf-8'))


def read_crop_resize(path, x1, y1, x2, y2, label):
    image_data = tf.read_file(path)

    decoded = tf.image.decode_jpeg(image_data)

    float32_img = tf.image.convert_image_dtype(decoded, tf.float32, saturate=True)

    batched = tf.expand_dims(float32_img, axis=0)

    cropped = tf.image.crop_to_bounding_box(
        batched,
        y1,
        x1,
        tf.maximum(1, y2 - y1),
        tf.maximum(1, x2 - x1)
    )

    resized = tf.image.resize_bilinear(cropped, (300, 200))

    # TODO Somewhere around here we need to turn the label into a 1-hot vec

    return resized, label, path


def encode_and_save(image, filename):
    image_int = tf.image.convert_image_dtype(image[0], tf.uint8, saturate=True)
    encoded_jpg = tf.image.encode_jpeg(image_int)
    return tf.write_file(filename, encoded_jpg)


def serialize(image, label, one_hot, path):
    image_string = tf.serialize_tensor(image)
    one_hot_string = tf.serialize_tensor(one_hot)

    return image_string, label, one_hot_string, path


def make_example(image_string, label, one_hot_string, image_path):
    return tf.train.Example(
        features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
            'one_hot': tf.train.Feature(bytes_list=tf.train.BytesList(value=[one_hot_string])),
            'image_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path])),
        })
    )


def all_labels(all_annotations: List[Tuple[str, int, int, int, int, str]]) -> List[str]:
    label_set = set()

    for _, _, _, _, _, label in all_annotations:
        label_set.add(label.strip())

    label_list = list(label_set)
    label_list.sort()

    return label_list


def make_one_hot(all_labels: List[str], label: tf.Tensor):
    get_index_op = tf.py_func(all_labels.index, [label], tf.int64)
    return tf.one_hot(get_index_op, len(all_labels))


def parse_test_set(dir) -> List[Tuple[str, str]]:
    """
    Parse the test directory to get file paths and labels.

    :param dir: The full path to the test set
    :return: a List[(path, label)] where path and label are str
    """
    output: List[(str, str)] = []
    for e in os.listdir(dir):
        if not e.startswith("."):
            name, _ = os.path.splitext(e)
            full_path = os.path.join(dir, e)

            # Split on '_', remove the last element (the number) and rejoin
            label = '_'.join(name.split('_')[:-1])
            im = Image.open(full_path)
            try:
                im.verify()
                output += [(full_path, label)]
            except Exception:
                continue

    return output


def process(annotations_path: str, output_train_path: str, output_test_path: str) -> None:
    all_annotations = list(files_gen(annotations_path))

    labels_list = all_labels(all_annotations)

    # Create a dataset of filenames, bounding boxes, and string labels
    filenames = tf.data.Dataset.from_generator(
        lambda: (i for i in all_annotations),
        (tf.string,
         tf.int32,
         tf.int32,
         tf.int32,
         tf.int32,
         tf.string)
    )  # type: tf.data.Dataset

    # Read and crop them on EIGHT THREADS HECK YEAH
    image_iter = filenames \
        .map(read_crop_resize, num_parallel_calls=8) \
        .map(lambda img_, label_, path_: (img_, label_, make_one_hot(labels_list, label_), path_), num_parallel_calls=8) \
        .map(serialize, num_parallel_calls=8) \
        .make_one_shot_iterator()  # type: tf.data.Iterator

    img_op, label_op, one_hot_op, path_op = image_iter.get_next()

    bar = ProgressBar(len(all_annotations))
    with tf.Session() as sess:
        writer_train = tf.python_io.TFRecordWriter(output_train_path)
        writer_test = tf.python_io.TFRecordWriter(output_test_path)
        while True:
            try:
                # This just writes the files back to disk.
                # TODO: serialize the processed images to TF/numpy binary format and save those along with the labels
                # op = encode_and_save(img, output_path + "/" + label + str(i) + ".jpg")
                img, label, one_hot_, path = sess.run([img_op, label_op, one_hot_op, path_op])
                serialized = make_example(img, label, one_hot_, path).SerializeToString()
                if random.random() < 0.9:
                    writer_train.write(serialized)
                else:
                    writer_test.write(serialized)
                bar.incr()
                bar.display()
            except tf.errors.OutOfRangeError:
                break

        writer_train.flush()
        writer_train.close()
        writer_test.flush()
        writer_test.close()


if __name__ == "__main__":
    annotations_path = sys.argv[1]  # Path to annotation.txt
    output_train_path = sys.argv[2]  # Path to output location for training
    output_test_path = sys.argv[3]  # Path to output location for testing

    process(annotations_path, output_train_path, output_test_path)
