from psmcnn import ps_mcnn
import argparse
from utils import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""parsing and configuration"""


def parse_args():
    desc = "Tensorflow implementation of PS-MCNN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='celebA', help='[celebA,LFWA]')

    parser.add_argument('--epoch', type=int, default=5, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch per gpu')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        cnn = ps_mcnn(sess, args)

        # build graph
        cnn.build_model()

        # show network architecture
        # show_all_variables()

        if args.phase == 'train':
            # launch the graph in a session
            cnn.train()

            print(" [*] Training finished! \n")

            cnn.test()
            print(" [*] Test finished!")

        if args.phase == 'test':
            cnn.test()
            print(" [*] Test finished!")


if __name__ == '__main__':
    main()