import argparse

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datastore_name', required=True, type=str, help='Name of the mounted Datastore')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=125)
    parser.add_argument('--val_epochs', type=int, help='epochs interval between each val_epoch', default=5)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--lr', type=float, default=9.5E-06, help='learning rate, default=0.0000095')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of the validation set, default=0.1')
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint')
    return parser.parse_known_args()