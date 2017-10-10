# Variance of activations as a proxy metric for network size

|Start Date|End Date  |
|----------|----------|
|2017-10-06|2016-10-08|

## Description

The goal here is to try to evaluate the distributions of the activation on MNIST dataset on a a clearly oversized network and see if variance is a good indicator of which neurons should be factored in the next layers.

The architecture will have only one (oversized) hidden layer.

## Delivrables

- [x] Pytorch code with the architecture and the training process
  - `/models/MNIST_h1.py` is the model
  - `/variance_metric.py` is the training procedure, and generates plots
- [x] Intepretation (+plots) of the variance on different network size
- [x] Conclusion on the value of the variance as indicator of network size


## Interpretation

As we can see on the plot, the bigger the network, the more the distribution of the variance of activation shift towards 0. This means that the more we add neurons, the more some of them are useless and can be factored in the next layer.

![Distribution of Activations](/plots/MNIST_1h_dist_activations.png?raw=true "Distribution of Activations")

Our goal here is to find wether the variance of activation can tell us if the network is oversized. Clearly 2048 is oversized and 64 is not but let's try to define the boundary between the two situations. Intuitively it would make sense to expect a normal distribution for the variance of the activation. As we can see on the results of the Shapiro normality tests, it seems that there is a sweet spot where the variance of the activation are almost normally distributed.

![Results of normality test](/plots/MNIST_1h_normality_test.png?raw=true "Results of normality tests")

The most normal distribution and the two neighbours are shown on a figure.

![Distribution of variance for the most normal network and the two neighbours](/plots/MNIST_1h_dist_activations_around_sweet.png?raw=true "Distribution of variance for the most normal network and the two neighbours")

We can also conjecture that the total amount of variance is
- ~~Either constant as we can see on the plot~~ (Better simulations invalidate this hypothesis)
- Logarithmic as what all ~~except two observations suggest~~(With multiple runs we obtain a perfect logarithm). ~~It is possible that the two outliers (1 and 3 from the end) are wrong because 5 epochs were not enough to train that many neurons~~. After training longer it seems that indeed the total amount of variance follows a logarithmic trend.

![Total amount of variance](/plots/MNIST_1h_sum_variance.png?raw=true "Total amount of variance")

We also see that the number of dead neurons (no variance at all) increase with the network size (alsmot linearly after some time).

![Number of dead neurons](/plots/MNIST_1h_dead_neurons.png?raw=true "Number of dead neurons")

## Conclusion

It seems that the variance of the activation of the hidden layer gives a lot of information about the size of the network. It is easy to see when a netowrk is oversized (when there are more dead neurons that active neurons). However it is hard to know what is the real boundary. The normality test seems to be promising but we will need to see if this metric agrees with the testing accuracy.


# Find relationship between the distribution of activations and the accuracy

|Start Date|End Date  |
|----------|----------|
|2017-10-08|2017-10-08|

## Description

The goal of this milestone is to find if there is a relation between the variance of the activation and the accuracy. It would be nice to see if it can help predicting when overfitting is happening. If such a thing exists then the variance of the activation would be a good candidate to control the shriking/expansion process.

## Delivrables

- [x] Pytorch code that evaluates the testing accuracy for each model
  - `variance_metric.py` was updated
- [x] Plots comparing the activations and the accuracy
- [x] Conclusion on the value of the variance as a control parameter for the network resize procedure

## Conclusion

![Comparison accuracy/normality](/plots/MNIST_1h_acc_vs_shapiro.png?raw=true "Comparison accuracy/normality")

As we can see from the plot, it seems that the normality of the variance of activations seems to be related. They reach their maximum around the same values (normality test yields on average a higher argmax). The two curves are quite noisy, it might be a good thing to rerun the experiments multiple time and aggregate the results.

In any case the variance of the activation seems to be a good candidate to control the shrinking and expansion process of a neural network. It seems to be a good proxy for accuracy but has the advantage that it gives information on each layer while the accuracy is only for the entire network. That means we can use it to resize each layer independently.

## Description

In the previous experiments, the values measured were sometimes noisy and probably partially random. It would be a good thing to run many training with different seeeds to make sure that the results are reproducible.

## Delivrables

- [x] Get access to GPUs that are accessible through ssh (Used them physically to save time)
- [x] Update existing code to run on CUDA
  - `variance_metric.py` now uses cuda
- [x] Pytorch code that run all the previous experiments multiple time and average the results
  - On titan Xp we were able to run 11 models in parallel.
- [x] Update all the plots with the aggregated measurements
  - All plots were updated
- [x] Global conclusion on the variance of activation
- [x] Propose an training algorithm that use this metric for resizing the layers
  - While the normality test seems to be able tell when a netowrk is properly sized, it seems complicated to use it to design a training algorithm using it. Indeed, just by looking at the t value there is not enough information to know if we can improve it by changing the size of the network, even worse, we don't know if we should shrink or expand the layer to get a better model.
  
# Conclusion

The distribution of the variance of activation seems to be a good proxy for network size. However the normality test (Shapiro) does not seem to be good enough since it suffers from two issues:
- Does not tell whether we should increase or decrease the number of neurons
- On average overestimate the optimal size of a network

# Investigate mixture models

|Start Date|End Date  |
|----------|----------|
|2017-10-09|          |

## Description

If we look carefully at the distribution of the variance of activations we can see that oversized network almost follow an exponential distribution. On the other side, undersized network follow a distribution similar to Poisson. In the middle they seem to be a mix of the two distribution. If we were able to estimate the parameter of both of the distribution and estimate the population size allocated to each distributions we would be able to tell accurately if we should shrink or expand a network.

## Delivrables

- [ ] Read litterature about Mixture models:
  - [x] [Wikipedia](https://en.wikipedia.org/wiki/Mixture_model)
  - [ ] Find books/lecture notes
- [ ] Try to implement an algorithm that separate the two components of the distribution
- [ ] Interpret and conclude



# See if observations extrapolate on the Zalando MNIST dataset

|Start Date|End Date  |
|----------|----------|
|          |          |

## Description

The goal of this milestone is to find if whatever we observed in the previous experiments if they generalize on another Dataset. The Zalondo MNIST here is a very good candidate since it has the exact same shape as the classic MNIST. It would make this experiment very interesting because the only factor changing is the distribution of inputs and outputs. Everything else stays the same.

## Delivrables

- [ ] Pytorch code that run the same procedure on the Zalando MNIST Dataset
- [ ] Interpretation of the difference between the two datasets
- [ ] Conclusion on robustness of whatever was found previously

# Run Multiple trainings to confirm results

|Start Date|End Date  |
|----------|----------|
|2017-10-08|2017-10-09|

