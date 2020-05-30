import argparse
import transformations as ts
import opt_tc as tc
import numpy as np
from transformations import Transformer_non90
from data_loader import Data_Loader

def transform_data(data, trans):
    print("Total Classficiation Transforms: ", trans.n_transforms)
    trans_inds = np.tile(np.arange(trans.n_transforms), len(data))
    trans_data = trans.transform_batch(np.repeat(np.array(data), trans.n_transforms, axis=0), trans_inds)
    return trans_data, trans_inds

def load_trans_data(args, trans):
    dl = Data_Loader()
    x_train, x_test, y_test = dl.get_dataset(args.dataset, true_label=args.class_ind)
    print("Non Augmented Data Shape: ", x_train.shape)
    #DATA AUGMENTATION
    normal_data_transformer = Transformer_non90(0,0,30)
    transformations_inds_aug = np.tile(np.arange(normal_data_transformer.n_transforms), len(x_train))
    print("Num Data Augments: ", normal_data_transformer.n_transforms)
    x_train_aug = normal_data_transformer.transform_batch(np.repeat(x_train, normal_data_transformer.n_transforms, axis=0),
                                                           transformations_inds_aug)
    print("Augmented Data Shape ", x_train_aug.shape)

    x_train_trans, labels = transform_data(x_train_aug, trans)
    x_test_trans, _ = transform_data(x_test, trans)
    x_test_trans, x_train_trans = x_test_trans.transpose(0, 3, 1, 2), x_train_trans.transpose(0, 3, 1, 2)
    y_test = np.array(y_test) == args.class_ind
    return x_train_trans, x_test_trans, y_test


def train_anomaly_detector(args):
    transformer = ts.get_transformer(args.type_trans)
    x_train, x_test, y_test = load_trans_data(args, transformer)
    tc_obj = tc.TransClassifier(transformer.n_transforms, args)
    # print(torch)
    tc_obj.fit_trans_classifier(x_train, x_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wide Residual Networks')
    # Model options
    parser.add_argument('--depth', default=10, type=int)
    parser.add_argument('--widen-factor', default=4, type=int)

    # Training options
    parser.add_argument('--batch_size', default=288, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=16, type=int)

    # Trans options
    parser.add_argument('--type_trans', default='complicated', type=str)

    # CT options
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--m', default=1, type=float)
    parser.add_argument('--reg', default=True, type=bool)
    parser.add_argument('--eps', default=0, type=float)

    # Exp options
    parser.add_argument('--class_ind', default=1, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)
    args = parser.parse_args()

    for i in range(1):
        # args.class_ind = 5+i
        print("Dataset: CIFAR10")
        print("True Class:", args.class_ind)
        train_anomaly_detector(args)
