# fastdeepnets

The progress of the project can be followed in the in [__project journal__](https://github.com/mitdbg/fastdeepnets/blob/master/journal.md)(Chronological order).

# Tools

## Regression CLI

We provide a CLI to train regression models for people who do not want to spend the time building their own models and data loaders. The training set is expected to be a CSV file with `i + o` columns where `i` is the number of inputs and `o` is the number of outputs.

__Example__:

`python tools/fit.py tools/example_regression_data.csv /tmp/model_test.data --layers 2 --max_neurons 1000 --input_features 32 --output_features 1`

In order for it to work make sure the `dynnet` package is available, either install the package or add the root of this repository in your `PYTHONPATH`.

## Goal

The goal of this project is to try to "learn" the number of hidden units (neurons for fully connected networks and channels for CNNs), directly during training. The main motivation is to reduce the size of set of potential hyper-parameters and avoid overfitting by killing neurons while the network is trained.

## Context

This project is part of my (Guillaume Leclerc) Master thesis
