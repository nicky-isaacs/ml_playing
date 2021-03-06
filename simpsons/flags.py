import argparse
import tensorflow as tf

class Flags:
    def __init__(self,
                 prediction_output_base_dir: str,
                 max_steps: int,
                 training_data: str,
                 test_data: str,
                 learn_rate: float,
                 logs_dir: str,
                 summary_interval: int,
                 annotations: int,
                 mode: str,
                 test_set: str,
                 enable_gpu: bool,
                 ):
        self.prediction_output_base_dir = prediction_output_base_dir
        self.max_steps = max_steps
        self.training_data = training_data
        self.test_data = test_data,
        self.learn_rate = learn_rate
        self.logs_dir = logs_dir
        self.summary_interval = summary_interval
        self.annotations = annotations
        self.mode = mode
        self.test_set = test_set
        self.enable_gpu = enable_gpu


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

    parser.add_argument('--summary-interval',
                        dest='summary_interval',
                        type=int,
                        default=100,
                        help='how many training steps between tensorboard metrics are posted (default 100)')

    parser.add_argument('--annotations',
                        dest='annotations',
                        type=str,
                        help='path to the annotations file')

    parser.add_argument('--mode',
                        dest='mode',
                        type=str,
                        default=tf.estimator.ModeKeys.TRAIN,
                        help='tensorflow mode: train, eval, infer (default: train)')

    parser.add_argument('--test-set',
                        dest='test_set',
                        type=str,
                        default='',
                        help='full path to the test set directory')

    parser.add_argument('--enable-gpu',
                        dest='enable_gpu',
                        type=bool,
                        default=False,
                        help='enable GPU support (default: False')

    args = parser.parse_args()

    all_args = [
        args.prediction_output_base_dir,
        args.steps,
        args.training_data,
        args.test_data,
        args.logs_dir,
        args.learn_rate,
        args.summary_interval,
        args.annotations,
        args.mode,
        args.test_set,
        args.enable_gpu
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
        summary_interval=args.summary_interval,
        annotations=args.annotations,
        mode=args.mode,
        test_set=args.test_set,
        enable_gpu=args.enable_gpu,
    )
