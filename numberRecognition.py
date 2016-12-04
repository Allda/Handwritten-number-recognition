import argparse


def setup_parser():
    parser = argparse.ArgumentParser(description='Hand writen number '
                                                 ' recognition')
    main_group = parser.add_mutually_exclusive_group()
    main_group.add_argument('--train-classifier', action='store_true',
                            help='Train classifier')
    main_group.add_argument('--classify-mnist', action='store_true',
                            help='Classify MNIST database')
    main_group.add_argument('--classify-own', default=None, metavar='FILE',
                            help='Clasify own picture with handwritten '
                                 'numbers')
    parser.add_argument('--mnist', default=None, nargs='?',
                        help='Location of MNIST database. In case it is '
                             'missing, it will try to download database')

    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    print args
    if args.train_classifier:
        print 'train'
        # TODO: call training function and dump result to file
    elif args.classify_mnist:
        print 'clasify mnist'
        # TODO: call classification function for MNIST data
    elif args.classify_own:
        print 'classify own picture'
        print args.classify_own
        # TODO: call classification function for own image

if __name__ == '__main__':
    main()
