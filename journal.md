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

# Run Multiple trainings to confirm results

|Start Date|End Date  |
|----------|----------|
|2017-10-08|2017-10-09|


## Description

In the previous experiments, the values measured were sometimes noisy and probably partially random. It would be a good thing to run many training with different seeeds to make sure that the results are reproducible.

## Delivrables

- [x] Get access to GPUs that are accessible through ssh (Used them physically to save time)
- [x] Update existing code to run on CUDA
  - `variance_metric.py` now uses cuda
- [x] Pytorch code that run all the previous experiments multiple time and average the results
  - On titan Xp we were able to run 11 models in parallel for each networks size (220+ models in total)
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
|2017-10-09|2017-10-16|

## Description

If we look carefully at the distribution of the variance of activations we can see that oversized network almost follow an exponential distribution. On the other side, undersized network follow a distribution similar to Poisson. In the middle they seem to be a mix of the two distribution. If we were able to estimate the parameter of both of the distribution and estimate the population size allocated to each distributions we would be able to tell accurately if we should shrink or expand a network.

## Delivrables

- [x] Read litterature about Mixture models:
  - [x] [Wikipedia](https://en.wikipedia.org/wiki/Mixture_model)
  - [x] [Interesting article about EM algorithm applied to mixture models](http://www.waset.org/publications/2675)
  - [x] [Expectation–maximization algorithm](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm)
  - [x] Find books/lecture notes
- [x] Try to implement an algorithm that separate the two components of the distributio
  - The idea seems to be good, but it is extremely hard to fit the model.
  - Will try to implement EM algorithm
  - EM is very efficient on a mixture of exponnential and gaussian distribution, but it does not look like it is the best model for the data
  - I tried fitting two gamma distributions and they fit the data very well. The only problem now is that the algorithm I have for EM in this case is randomised and have poor convergence properties. ~~I will try to have a deterministic algorithm and the results should be even better (and much faster than the randomised one, based on bootstraping)~~ (I now have an efficient and stable algorithm)
  - Implemented two very efficient fitting algorithms based on the EM concept:
    - `./algorithms/exp_norm_mixture_fit.py` : that fits a mixture of an exponnential and a normal distribution
      - The fit obtained with this model is quite good but has a tendency to overestimate the amount of dead neurons
    - `./algorithms/digamma_mixture_fit.py`: that fits a mixture of two gamma distribution
      - It seems this one fits better but it has a disadvantage because it might fit two distribution that are gaussian-like when the network is properly sized
- [x] Generate plots
- [x] Interpret and conclude

# Interpretation 
![Comparison of different metrics](/plots/MNIST_1h_acc_vs_mixtures.png?raw=true "Comparison of different metrics")

As we can see on this plot, the thre variance-based metrics seem to moderately underestimate the size of the network, but they are still interesting for our purpose. The number of dead neurons might be promising too except that whatever the threshold we might set it would be a completely arbitrary value (except if we choose zero but it does not seem to be a good one).

In the next section we will try to see how it generalizes on an another data set.


# See if observations extrapolate on the Zalando MNIST dataset

|Start Date|End Date  |
|----------|----------|
|2017-10-16|2017-10-16|

## Description

The goal of this milestone is to find if whatever we observed in the previous experiments if they generalize on another Dataset. The Zalondo MNIST here is a very good candidate since it has the exact same shape as the classic MNIST. It would make this experiment very interesting because the only factor changing is the distribution of inputs and outputs. Everything else stays the same.

## Delivrables

- [x] Pytorch code that run the same procedure on the Zalando MNIST Dataset
  - `variance_metric.py` was updated
- [x] Interpretation of the difference between the two datasets
- [x] Conclusion on robustness of whatever was found previously

## Interpretation

Before we interpret any results it is important to understand why we choose to compare against the FashionMNIST instead any other machine learning task:

- FashionMNIST is a *clone* of the orginal dataset
  - Same number of inputs
  - Same domain (`[0;1]`) for inputs
  - Same number of classes
 - Therefore the model itself does not have to change. We are sure that we are not introducing any other external factors that could impact the results fo the comparison
 - It even has the same distribution of outputs. It means the variance on the outputs is the same in the two experiments.
 - The only difference between the two problems is their difficulty
 
 With this experiment we want to challenge the following thesis
 
 *The metrics designed in the first steps are useful in a sense that they can predict if a netowrk is undersized or oversized*
 
 Here by switching to FashionMNIST we are increasing the difficulty of the problem so this is what we are expecting
 - The accuracy plateau occurs later on the most complext dataset (The model needs more neurons to reach its optimal accuracy)
 - The distributions of the variance for a given number of neurons shoud be shifted on the right for the more complex dataset
 - The metrics indicating that the network is oversized also occurs later (The model predicts that we need more neurons for this problem.
 
 If the first hypothesis is true (we reach the plateau at roughly 1024 neurons for FashionMNIST vs 256 for MNIST). Unfortunately, the two others do not seem to hold. (Note that to ensure that the results were not training time dependant I doubled the training time from 15 to 30 epochs so this variable cannot explain the results).
 
 When we look at the distributions, we can see that in the undersized configuration (362 neurons for example) the average activation is higher on the simple dataset. In the oversized configuration (2896 neurons) the density of useless neurons is higher on the more complex task. If you think about it, this result makes sense. If you consider that a classification neural network tries to "separate" classes, when it is facing a hard task the boundaries it draws in the input space might not be well placed. As a result one class might get more elements than it should. Since the inputs are uniformly distributed then in this case the variance is necessarly lower in the more complex task.
  
 ![Distribution of Activations - MNIST](/plots/MNIST_1h_dist_activations.png?raw=true "Distribution of Activations - MNIST")

 ![Distribution of Activations - FashionMNIST](/plots/FashionMNIST_1h_dist_activations.png?raw=true "Distribution of Activations - FashionMNIST")

Since the assumption that the distribution of activation is a proxy for the size of the network, necessarly the last hypothesis does not hold (except for the number of dead neurons since it does not rely on the distribution of activations)

![MNIST Metrics](/plots/MNIST_1h_acc_vs_mixtures.png?raw=true "MNIST Metrics")

![FashionMNIST Metrics](/plots/FashionMNIST_1h_acc_vs_mixtures.png?raw=true "FashionMNIST Metrics")

## Conclusion

As we showed in this experiment the variance of activation does not seem to be a good candidate to estimate the size of a network in the general case. We introduced potential explnations of these counter intuitive results and could spend time trying to confirm  them or not but it would not change the concusion on the validity of the variance of activations as a metric so I don't think it is worth spending time on it.

In the near future we will have to investigate other metrics in order to get a general estimator of network size.

Candidates are now:

- Number of dead neurons (threshold seem to be arbitrary and past experiments shows that it underestimate the network size on FashionMNIST)
- PCA decomposition of the weight matrix
- Factorization of the weight matrix in two matrix that minimizes the size of the intermediate result (output compression similar to autoencoders)

# Investigate PCA based metrics for layer size

|Start Date|End Date  |
|----------|----------|
|2017-10-17|          |

## Description

The idea behind this experiment is that PCA gives you a kind of factorization that minimizes the latent space (you can choose the size of this space ant it will minimize the error introduced by this reduction). What we would like to do is to find for each layer, what is the real number of dimensions we need to express the data. This can be interpreted as the minimmal number of neurons that are needed to express this transformation (the layer). The key difference between the measurements and the ones with the variance is that we will have to hook before the activation function otherwise the values will not be nicely spread and we will see sharp boundaries.

## Delivrables

- [ ] Pytorch code that estimate the minimum viable dimensionality for a given model
- [ ] Interpretation 
- [ ] Conclusion

