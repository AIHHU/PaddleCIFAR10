import argparse

def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='resnet')

    parsed = parser.parse_args()
    options = parsed.__dict__

    return options