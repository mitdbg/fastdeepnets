# Variance of activations as a proxy metric for network size

|Start Date|End Date  |
|----------|----------|
|2017-10-06|          |

## Description

The goal here is to try to evaluate the distributions of the activation on MNIST dataset on a a clearly oversized network and see if variance is a good indicator of which neurons should be factored in the next layers.

The architecture will have only one (oversized) hidden layer.

## Delivrables

- [x] Pytorch code with the architecture and the training process
  - `/models/MNIST_h1.py` is the model
  - `/variance_metric.py` is the training procedure, and generates plots
- [x] Intepretation (+plots) of the variance on different network size
- [ ] Conclusion on the value of the variance as indicator of network size


## Interpretation

As we can see on the plot, the bigger the network, the more the distribution of the variance of activation shift towards 0. This means that the more we add neurons, the more some of them are useless and can be factored in the next layer. Our goal here is to find wether the variance of activation can tell us if the network is oversized. Clearly 2048 is oversized and 64 is not but let's try to define the boundary between the two situations. Intuitively it would make sense to expect a normal distribution for the variance of the activation. As we can see on the results of the Shapiro normality tests, it seems that there is a sweet spot where the variance of the activation are almost normally distributed. The most normal distribution and the two neighbours are shown on a figure.

We can also conjecture that the total amount of variance is
- Either constant as we can see on the plot
- Logarithmic as what all except two observations suggest. It is possible that the two outliers (1 and 3 from the end) are wrong because 5 epochs were not enough to train that many neurons

We also see that the number of dead neurons (no variance at all) increase with the network size (alsmot linearly after some time).

## Conclusion

It seems that the variance of the activation of the hidden layer gives a lot of information about the size of the network. It is easy to see when a netowrk is oversized (when there are more dead neurons that active neurons). However it is hard to know what is the real boundary. The normality test seems to be promising but we will need to see if this metric agrees with the testing accuracy.


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
