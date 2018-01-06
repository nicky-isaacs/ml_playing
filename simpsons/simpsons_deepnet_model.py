from typing import List

import tensorflow as tf

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

def conv_layer(
        input: tf.Tensor,
        filter_count: int,
        kernel_size: List[int],
):
    return tf.layers.conv2d(
        inputs=input,
        filters=filter_count,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)


def pooling_layer(
        input: tf.Tensor,
        pool_size: List[int],
        strides: int,
):
    return tf.layers.max_pooling2d(
        inputs=input,
        pool_size=pool_size,
        strides=strides)

def graph(
        layers: int,
        dropout_rate: float,
        num_classes: int,
        learn_rate: float,
        img_height: int,
        img_width: int,
        img_channels: int,
        training_hooks: List[tf.train.SessionRunHook],
        model_dir: str,
        save_steps: int,
):
    """Intended to be used with tf.estimator.Estimator. Returns a function which adheres to the Estimator
    model_fn param (accepts: features, labels, mode, params, config)

    :param layers: the number of layers in the model. must be >= 1. Each layer consists of a convolution and pooling step
    :param dropout_rate:
    :param num_classes:
    :param learn_rate:
    :param img_height:
    :param img_width:
    :param img_channels:
    :return: A model fn
    """

    def model_fn(
            features,
            labels,
            mode,
            params,
            config,):
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        input_layer: tf.Tensor = tf.reshape(features, [-1, img_width, img_height, img_channels])

        # For every layer, create a convolution with Nx10 + 10 filters (10, 20, 30, etc.) and a max pool
        # of 2x2 with stride 2
        for i in range(layers):
            with tf.name_scope(f"layer_{i}"):
                filters = (i * 10) + 10
                conv = conv_layer(input_layer, filters, [5, 5])
                input_layer = pooling_layer(conv, [2, 2], 2)


        flattened = tf.layers.flatten(input_layer)

        # Dense Layer
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 1024]
        with tf.name_scope("dense"):
            dense = tf.layers.dense(inputs=flattened, units=1024, activation=tf.nn.relu)

        # Add dropout operation; 0.6 probability that element will be kept
        with tf.name_scope("dropout"):
            dropout = tf.layers.dropout(
                inputs=dense, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, num_classes]
        with tf.name_scope('logits'):
            logits = tf.layers.dense(inputs=dropout, units=num_classes)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels =  tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits)

        tf.summary.scalar("loss", loss)

        _, accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

        tf.summary.scalar("accuracy", accuracy)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            saver_hook=tf.train.SummarySaverHook(
                summary_op=tf.summary.merge_all(),
                save_steps=save_steps,
                output_dir=model_dir
            )

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=training_hooks + [saver_hook]
            )

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return model_fn
