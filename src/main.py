#!/usr/bin/env python3

# from utils.feature_extraction import save_extracted_features
from src.config import train_test_dir
from os.path import join
from numpy import load


def main():
    X = load(join(train_test_dir, 'sparsify/X.npy'))
    print(X.shape)

    Y = load(join(train_test_dir, 'sparsify/X.npy'))
    print(Y.shape)
    pass


if __name__ == "__main__":
    main()
