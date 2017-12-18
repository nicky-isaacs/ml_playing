import csv
import os

import tensorflow as tf
import sys


def files_gen(annotations_path: str):
    with open(annotations_path, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            relative_path = row[0]
            csv_base_path = os.path.dirname(annotations_path)
            image_path = os.path.abspath(os.path.join(csv_base_path, relative_path))
            ret = (image_path,
                   int(row[1]), int(row[2]),  # upper left x,y
                   int(row[3]), int(row[4]),  # lower right x,y
                   row[5])
            print(ret)
            yield ret


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

    return resized, label


def encode_and_save(image, filename):
    image_int = tf.image.convert_image_dtype(image[0], tf.uint8, saturate=True)
    encoded_jpg = tf.image.encode_jpeg(image_int)
    return tf.write_file(filename, encoded_jpg)


def process(annotations_path, output_path):
    # Create a dataset of filenames, bounding boxes, and string labels
    filenames = tf.data.Dataset.from_generator(
        lambda: files_gen(annotations_path),
        (tf.string,
         tf.int32, tf.int32,
         tf.int32, tf.int32,
         tf.string)
    )  # type: tf.data.Dataset

    # Read and crop them on EIGHT THREADS HECK YEAH
    image_iter = filenames \
        .map(read_crop_resize, num_parallel_calls=8) \
        .make_one_shot_iterator()  # type: tf.data.Iterator

    img, label = image_iter.get_next()

    i = 0
    sess = tf.Session()
    while True:
        try:
            # This just writes the files back to disk.
            # TODO: serialize the processed images to TF/numpy binary format and save those along with the labels
            op = encode_and_save(img, output_path + "/" + label + str(i) + ".jpg")
            sess.run(op)
            i += 1
        except tf.errors.OutOfRangeError:
            break


if __name__ == "__main__":
    annotations_path = sys.argv[1]  # Path to annotation.txt
    output_path = sys.argv[2]  # Path to output location

    process(annotations_path, output_path)
