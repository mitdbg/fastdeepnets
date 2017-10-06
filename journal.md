# Variance of activations as a proxy metric for network size

|Start Date|End Date  |
|----------|----------|
|2017-10-06|          |

## Description

The goal here is to try to evaluate the distributions of the activation on MNIST dataset on a a clearly oversized network and see if variance is a good indicator of which neurons should be factored in the next layers.

The architecture will have only one (oversized) hidden layer.

## Delivrables

- [x] Pytorch code with the architecture and the training process
- [ ] Intepretation (+plots) of the variance on different network size
- [ ] Conclusion on the value of the variance as indicator of network size

# Find relationship between the distribution of activations and the accuracy

|Start Date|End Date  |
|----------|----------|
|          |          |

## Description

The goal of this milestone is to find if there is a relation between the variance of the activation and the accuracy. It would be nice to see if it can help predicting when overfitting is happening. If such a thing exists then the variance of the activation would be a good candidate to control the shriking/expansion process.

## Delibrables

- [ ] Pytorch code that evaluates the testing accuracy for each model
- [ ] Plots comparing the activations and the accuracy
- [ ] Conclusion on the value of the variance as a control parameter for the network resize procedure
