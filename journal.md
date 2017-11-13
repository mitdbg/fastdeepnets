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

![Distribution of Activations](/plots/MNIST_1h_dist_activations.png?ra=true "Distribution of Activations")

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
|2017-10-17|2017-10-20|

## Description

The idea behind this experiment is that PCA gives you a kind of factorization that minimizes the latent space (you can choose the size of this space ant it will minimize the error introduced by this reduction). What we would like to do is to find for each layer, what is the real number of dimensions we need to express the data. This can be interpreted as the minimmal number of neurons that are needed to express this transformation (the layer). The key difference between the measurements and the ones with the variance is that we will have to hook before the activation function otherwise the values will not be nicely spread and we will see sharp boundaries.

## Delivrables

- [x] Pytorch code that estimate the minimum viable dimensionality for a given model
  - New file `pca_metric.py` has been added. It performs CPU PCA and GPU processing of the results and then generates the plots
- [x] Interpretation 
- [x] Conclusion

# Interpretation

For this task we decided to measure before the activation. The main reason behind this decision is that since we are doing a PCA, it will not interact well with the domain being "cut" by the boundary of the activation (here, ReLU). We can see on the following figure the explained variance of each PCA component sorted by their importance. We did the same plot for the two dataset


![MNIST PCA explained variance](/plots/MNIST_1h_pca_explained_variance.png?raw=true "MNIST PCA explained variance")

![FashionMNIST PCA explained variance](/plots/FashionMNIST_1h_pca_explained_variance.png?raw=true "FashionMNIST PCA explained variance")

By looking at these two plots it hard to see anything really significant. It seems that the tangent at the last point before it "dives". But if we pay close attention to the FashionMNIST we see that there are steep tips for small networks, then it attenuates and start again. Nothing in these plot could explain the difference between the 200 neurons required for the simple dataset and the 500+ for the more complex one
 
The second metric we investigate is the reconstruction error. We measure the average distance between a data point and the same with itself without only k components of the PCA. Here are the results:

![MNIST PCA reconstruction distance](/plots/MNIST_1h_pca_reconstruction_distance.png?raw=true "MNIST PCA reconstruction distance")

We would suspect that there would be some components more important than others. But as this plot suggests, all components are equally important. However it seems that the slope of the curve might have some signification. Lines seems to be parallel to the last one. Using this metric the optimal size would be arround 350 neurons. Now let's take a look at the same plot for the complex dataset.

![FashionMNIST PCA reconstruction distance](/plots/FashionMNIST_1h_pca_reconstruction_distance.png?raw=true "FashionMNIST PCA reconstruction distance")

Here the slope seem to be just noise and there is nothing we can clearly see

## Conclusion

It really looks like there is nothing we can extract from all these metric that are based on the PCA of the activations. The two first approach we tried are inconclusive. We can start doubting that the goal of trying to find a metric that takes information about a layer and tells wether it is properly sized is not vain. It leads us to this conjecture:

### Conjecture

__*There is no metric that takes information about a layer and determine if it is oversized, undersized or properly sized*__

### "Proof"

Let's assume that such a function `f` exists and works. We take a dataset `d1` and train a model `M` with a single layer. We choose the number of neurons such that the answer `f(M)` is `undersized`. Now we consider a new dataset `d2`. Its inputs are random noise of the same shape as the inputs of `d1`, and the outputs are `M(d1_inputs)`. By construction the accuracy is 100%, therefore it is properly sized. `M` has not changed therefore `f(M)` is still `undersized` therefore `f` does not work because it is wrong on the second dataset. It can't be undersized sine we have a 100% accuracy on the second dataest. 

# Implement a metric-less resizing

## Description

Since it seems that finding a proper metric is very hard and might be vain, during this project we will try to implement a resizing procedure that is automatic (it is not driven by a metric to know whether we should increase or decrease the size of a layer).


|Start Date|End Date  |
|----------|----------|
|2017-10-23|2017-10-23|

## Delivrables

- [x] Design and formalize a metric-less training procedure
  - Algorithm is the following
    ```python
    
    # Hyper Parameters
    
    l = 0.9 # Ratio of the real loss (for the multi objective optimization)
    
    ## Initialization
    
    running_mean = zeros(N)
    running_var = zeros(0)
    bias = zeros(N)
    k = 1
    x_0 = N
    
    ## For each batch
    
    x = W @ x + bias  # Compute the output of the layer
    x = act(x) # go through the activation function
    factors = 1 -  1 / (1 + exp(-k * (range(0, N) - x_0))) # Compute the scaler for each neuron
    x = x * factors
    return x
    
    ## compute the loss
    
    loss = original_loss
    for k, x_0 in layers:
      loss += l * sum(1 -  1 / (1 + exp(-k * (range(0, N) - x_0))))
    ```
  - I don't think this will work as is because the low variance neurons might be at the begining
- [x] Implementation in pytorch
  - `/models/MNIST_1h_flexible.py` was added. It contains a model with the scaler applied
  - `/simple_metricless.py`. It contains the training procedure and integrate the loss
- [x] Interpret the results
  - The gradient of the parameters `k` and `x_0` are `Nan`. Probably because the way it was computed is pretty messy. There are no results with `Nan` values in gradients
- [x] Conclude
  - We should implement the gradient calculation ourselves instead of relying on `torch.autograd` since it fails miserably here.


# Improve numerical stability

## Description

As we saw in the previous implementation we are only getting NaNs in the gradients. We need to find a way to fix it and make the training procedure numerically stable

|Start Date|End Date  |
|----------|----------|
|2017-10-24|2017-10-30|

## Delivrables

- [x] Find the integral of the scaler instead of doing the sum.
- [x] Find a way to compute it in a stable way (avoid inf and NaNs)
- [x] Update the training procedure
- [x] Generate plots
  - Plot generation was added in `/simple_metricless.py`
- [x] Interpret
- [x] Conclude

## Interpretation

Pr Madden suggested that the entire integral calculation was pointless and could just be replaced with the `x_0` parameter. It did make a lot of sense but after trying I noticed that the system diverged more often than it should. After investigation the reason is pretty simple. The integral tend to `0` when `x_0` tend towards -infinity and `size` when infinity. By replacing the integral by `x_0` we allow the loss to be as low as -infinity which does not make sense. The loss associated to the network size should always be positive.
 
 As we can see on the plots whatever the starting size is the NN always converge (roughly) to the same number of neurons. This is great news because that shows that the function is smooth enough not to get stuck in local minima.
 
![MNIST learned network size](plots/MNIST_1h_simple_flexible_convergence.png?raw=true "MNIST learned network size")
 
We can also make the same observation for the accuracy. It does not depend on the starting network size.

![MNIST accuracies](plots/MNIST_1h_simple_flexible_accuracies.png?raw=true "MNIST accuracies")

Now it would be interesting to compare the accuracy obtained with
  - a flexible network against a static network trained with the size the flexible network converged to
  - The best accuracy available
  
We did that on the **testing accuracy** and here are the results:
  
![MNIST accuracy comparaison](plots/MNIST_1h_simple_flexible_frontier.png?raw=true "MNIST accuracy comparaison")
  
![FashionMNIST learned network size](plots/FashionMNIST_1h_simple_flexible_frontier.png?raw=true "FashionMNIST learned network size")

There are key observations to make from these plots:

- The loss in accuracy is much higher on `FasionMNIST` than `MNIST`
- The NN size seems to be converging to some plateau
- On `MNIST` the flexible model seems to be as good as the static model. It is not the case at all on `FashionMNIST`
- The size `MNIST` converge to is higher than the size `FashionMNIST` does. It is counter intuitive because the former is simpler than the latter.

## Conclusion

Even if the results obtained are interesting there a re a lot of question to be answered

- Why does the size converges to a plateau ? Why can't we reach arbitrary big networks ?
- Why `MNIST` is bigger than `FashionMNIST`
- Can we reach the same accurcy on the flexible network than the static one ?

We will try to answer these question later in this journal.

# Investigate the origin of the network size plateau and try to get rid of it

## Description

As we can see on both `MNIST` and `FashionMNIST` it seems the "optimal" network size converges to some problem specific value. For example for `MNIST` it seems impossible to use more than 70 neurons. However we know that we can do better accuracies with a slightly bigger network. The question here is why do we hit this wall and can we comfigure the trainign algorithm to use more than 70 neurons and reach the best accurcy.

|Start Date|End Date  |
|----------|----------|
|2017-11-01|2017-11-01|

## Delivrables

- [x] An explanation/intuition about the origin of this plateau
  - Analysis below
  - Plots generated by `validate_plateau_hypothesis` in `/simple_metricless.py`
- [x] Algorithm to overcome this plateau
- [x] Conclusion

## Analysis

### Observations
- Even if we remove the size of the network from the loss, the size of the network converges to the same value (the same value as if the weight is very small, ie: `10^-8`)
- It seems it depends on the l2 regularization that we apply on the weight of the linear layers. It seems as if the size of the network was accounted for in the l2 norm, even if the code says it should not

### Hypothesis

The lower `x_0` is the more outpus are zeros. When outputs are zero then the lower layers can afford setting their weights to zero for these neurons. It reduces the total l2 norm of the system and have a lower penalty. Increasing the size of `x_0` would require to set non zero weights to use them and increase the l2 norm

### Verifying the hypothesis

To verify our hypothesis we will train a flexible network with usual parameters and we will look at the weights in the last layer that are associated with each neuron of the hidden layer. If they are all zero then our hypothesis make sense.

As we can see on this plot. This hypotesis is clearly valid. All the weigths associated to disabled neurons are zero. The l2 regularization is without any doubt the reason why this happens.

![Netowrk size plateau size explanation](plots/MNIST_1h_plateau_explanation.png?raw=true "Network size plateau explanation")

To improve the confidence in our hypothesis we do another experiment and look at the same plot without any l2 regularization.
We can see on the second plot that the weights are not zero anymore and are as noisy as they were at the beginning of training

![Associated weights without l2 penalty](plots/MNIST_1h_plateau_explanation_no_pen.png?raw=true "Associated weights without l2 penalty")

## Potential algorithms
- As we saw in the previous step. The culprit is the l2 regularization factor in the loss function. If we get rid of it then it should work. Before removing it I think it is important to understand why we put it in there in the first place:
  - Before we wanted to train the slope of the sigmoid
  - For very low slopes there would be a lot of neurons in the "gray area" which means they have a scaling factor in `]0, 1[`
  - In this open interval downstream neurons can compensate this values by increasing their weights
  - That means every neuron in the "gray area" can turn into a fully functional normal neuron.
  - Adding a l2 regularization forbid downstream neurons to have incredibly large weight that counterbalance the scaling factor
  
We think that from here we have two different options we might want to follow:
  - **Get rid of the l2 regularization factor entirely**: This might be a good thing but we want to make sure that the weights in the lower levels are not exploding. The reason it might actually work is because now that we settled on a fixed slope. It is imossible to converge to a situation where the scaler has a lot of small values (therefore small total integral) that are compoensated by non-punished weights in downstream layers.
  - **Have very small regularization factor**: We want to have a regularization factor that is so small that it have no influence  on the size of the network but forbid completely stupid (10^...) weights

## Conclusion

During this task we have explained the presence of this plateau and proposed multiple alternatives to overcome it. This is very promising in a sense that using this very simple flexible model we still have a possible chance to reach the same accuracies as static models.

# Analyse the behavior of flexible netowkrs without L2 regularization

## Description

As we saw in the previous task getting rid of the L2 regularization might be a potential solution to converge to bigger (even maybe optimal network sizes). We can test this hypothesis and make sure that they behave properly.

|Start Date|End Date  |
|----------|----------|
|2017-11-01|2017-11-02|

## Delivrables

- [x] Analyse the convergence properties of flexible networks without regularization
- [x] Analyse the weights of underlying layers to make sure that nothing weird is going on
- [x] Conclude about this method

## Interpretation

As we can see from these plots (__these plots are on the training set and not testing unlike the previous measurements__), The performance of models without the L2 regularization are much better. First, they converge to bigger yet reasonable size. And we can see that flexible netowrks are extremely competitives with static networks. They are still underperforming on `FasionMNIST`

![MNIST - Performance with different penalties without l2 reg and on training set](plots/MNIST_1h_simple_flexible_frontier_without_penalty_training.png?raw=true "MNIST - Performance with different penalties without l2 reg and on training")

![FashionMNIST - Performance with different penalties without l2 reg and on training set](plots/FashionMNIST_1h_simple_flexible_frontier_without_penalty_training.png?raw=true "FashionMNIST - Performance with different penalties without l2 reg and on training")

We still observe the two interesting properties:

- Convergence to the same value regardless of the initial size
  - ![MNIST - size convergence without l2 reg](plots/MNIST_1h_simple_flexible_convergence_without_penalty_training.png?raw=true "MNIST - size convergence without l2 reg")
- Convergence to the same accuracy regardless of the initial size
  - ![MNIST - accuracy convergence without l2 reg](plots/MNIST_1h_simple_flexible_accuracies_without_penalty_training.png?raw=true "MNIST - accuracies convergence without l2 reg")
  
## Conclusion

We can see tha removing the L2 regularization improved the performance of the flexible layer. However still some questions remains:

  - Why do we still have a size plateau (2x bigger now)
  - Why `FashionMNIST` converges to lower size than `MNIST`
  
We will have to answer these questions to be able to improve again this model.

## Erratum

__There was a huge error in the implementation of flexible networks. The starting size of the network was ignored__

The new conclusion is that now the network is not converging to the same size for any size. In every case the size increases (which is what we would expect because there is no penalty on the size of the network) but eventually it get stuck in a local mimimum. A potential explaination is that the left part was trained therefore reducing the size of the newtork would have a bad impact on the loss and the right part was never activated therefore is still random. Eventually we will reach a point where adding neurons from the right part (random neurons) will also have a bad impact on the loss. Therefore the size does not change anymore.

As we can see from the plot it seems that there is no correlation (even maybe an anti-correlation) between the starting size and the end accuracy. One possible explanation is that 15 epochs is not enough to train the larger netowrks. __More experiments will be needed to validate this hypothesis__

# Try to get rid of the local minima

## Description

For every starting  point we eventually end up in a situation where the right part of the network has never been trained and the left part is well trained. When it occurs the size of the network is stuck in this local minimal and never changes.

In this step we would like to mitiage or get rid of this problem and have the network tend to an infinite size (or very large) when there is no penalty applied on the size of the network.

There are two potential solutions to this problem right now:

## Random Slope

### Pros:

- Easy to implement
- Lightweight calculations
- Similar to a dropout on a subset of the neurons. It might actually help generalization

### Cons:

- Results may vary a lot depending on the distribution
- The possible values for the slope need to be bounded when we implement the optimized version (variable size matrices)

### Interpretation and conclusion

First let's take a look at the experiment we made:

- Deterministic Model with a slope of `1` ("gray zone" approximatly 10)
- Model with "gray zone" size uniformly distributed between 1 and 20, which meanse the expected "gray zone" is the same as the Deterministic model
- Train 30 models of each to reduce the importance of the initialization
- Train for 30 epochs
- Observe the size of the network, the average gradient of the size per epoch and the average batch loss
- For `MNIST` and `FashionMNIS`

Now let's look at the results

#### MNIST

![MNIST - comparison with random slope model](plots/MNIST_1h_deterministic_random_comparison.png?raw=true "MNIST - comparison with random slope model")

#### FashionMNIST

![FashionMNIST - comparison with random slope model](plots/FashionMNIST_1h_deterministic_random_comparison.png?raw=true "FashionMNIST - comparison with random slope model")

We can see couple of very interesting things on these plots:

- Unlike we thought earlier the size of the network does not converge to some value but diverges in a logarithmic fashion.
- Having random slope does not seem to be a greater idea. It makes the gradient more noisy and converge to infinity slower. Moreover loss is always higher.

## Random actiavtion

### Pros

Same as the Random slope

### Cons

No obvious cons except that it might not help at all the local minimal problem

### Conclusion

With this method gradients will always be zero so this is completely unusable.

## Two phase training

The idea behind two phase training is:

- First we train the "normal" part of the network
- We train the right part (gray zone + a little area) with a higher learning rate

The goal is to avoid having garbage on the right side of the network

### Pros:

- Very likely that it solves the problem

### Cons:

- Two times more calculations (two forward and two backward passes)
- Tedious to implement

### Pre-Task - Compare evolution of size with a pretrained network

Before going into the implementation of this method, there is a quick test we can do to check if there is a chance it will actually work. If we train a network until convergence and then we restart training with a size starting at zero, we should see the size increasing without any problem. This is a simpler version of the technique we propose here. Indeed, if the second part of the trianing was perfect in one iteration then we would end up in the same situation.

#### Results of the pretask

As we can see from the plots below:

- The more we train the network before enabling the variable size, the lower the size it converges to
  - One possible explanation to this is that: since it is easier to increase the activation of neurons by activating them than training the weights it is possible that the having pretrained weights reduce the need to tap into the new pool of neurons
- The three netowrk eventually end up in the same state and the same loss on average. This is good because it means they can reach the same loss with fewer neurons
- We observe high amount of noise after 30 epchs. Two explanations come to my mind:
  - We reach the weird state of Adam optimizer when the loss start exploding when it is too low. Since we are averaging 30 neurons here the spikes in the loss might be hidden by the high number of experiments
  - After a lot of training the network might be very susceptible to change in the number of neuron and since the learning rate for the number of neuron is much migher than the one for the weights, it is possible that it cause the noise.
- Pretraining the netowrk converges to lower network size. It means that trying the right side faster will not help reaching infinite netowrk size.

### Conclusion on the two phase training

All these experiments brought a lots of insights but sadly this method will not be able to help as we saw in the Pre-Task. As a result we will not spend time implementing it.
  
#### MNIST Plots

![MNIST - comparison with pretrained network](plots/MNIST_1h_flexible_behavior_on_pretrained.png?raw=true "MNIST - Comparision with pretrained network")

#### FashionMNIST Plots

__Being Rendered right now__

|Start Date|End Date  |
|----------|----------|
|2017-11-06|2017-11-08|

## Delivrables

- [x] Try the random activation method
  - __Gradient always zero__
  - Since this is pointless. Code was not even checked in the repository
- [x] Try the random slope method
  - Implemented in `models/MNIST_1h_flexible_random.py`
- [x] Try the two phase training method
- [x] Produce plots
- [x] Interpret
- [x] Conclude

## Interpretation

As we can see from these three potential methods. None of them was able to solve the plateau problem.

## Conclusion

During these unsuccessful attempts at breaking the plateau. I thought of another factor that might be causing this problem. As we can see on the the plots above the gradient on the size is sometimes positive (negative on the plots but we show the direction of movement instead of the gradient because we are minimizing the loss). And it does not make sense to have positive gradients here because with no constraint adding more neuron can only eventually reduce the training loss. The only reason these gradient would be positive is because there are neurons in the gray zone that should have their value reduced. In this case it might be better to change the size instead of changing the weights. A way of checking this would be to pretrain a network and then freeze the weights and only train the size of the netowrk. This hypothesis would also explain why `FashionMNIST` is smaller than `MNIST`. Indeed, sice `FashionMNIST` is a much harder problem then it is possible that the decision about neurons might be much more noisy and if at an epoch the average decision about the gray zone is lower the absolute value of these neurons then the size of the network will be reduced.

# Check if Adam is messing up with the training at very low loss

## Description

We saw repeteadly in previous experiments that after a large number of epochs. The loss starts to become very noisy. This is unexpected and we would like to understand why and try to solve it if possible


|Start Date|End Date  |
|----------|----------|
|          |          |

## Delivrables

- [ ] Plot the min and max loss of multiple models during training
- [ ] Compare with AdaMAX optimizer to see if that solves the problem
- [ ] Interpretation
- [ ] Conclusion

# Try training only the size on a thoroughly trained network

*to do later*

# Try to break the plateau

## Observations

As we saw in the previous experiments it seems reasonable to conjecture that the main problem of the technique is that the "gray zone" is compeltely arbitrary. It makes the optimization process extremely non-smooth and very important neurons might be at the right of the network. It has two consequences:

- Either we end up killing them and we have a lower accuracy than what we really need
- Either we keep them and as a result we have a size bigger than the optimal for the given problem

In any case the solution is sub-optimal. Another big problem is that the size is:

- __Starting point dependent__ (Especially when we have zero penalty)
- __Penalty dependent__. In a sense it may seem normal, because we want to specify the trade-off between size and accuracy
- __Problem dependent__. It is normal but in this case it is reversed. `FashionMNIST` ends up being smaller than `MNIST`

## Potential Solutions

I think the key to solve this problem is to make sure that we remove/insert the neurons in order of importance. I see two potential solutions:

### Sorting the neurons according to a specific metric

If we could do that that would mean that the gray-zone would contain the worst neurons (and potentially new neurons).

The perfect metric would be *The difference in loss if we were to disable this neuron*. The problem is that evaluating this metric is way too expensive (an entire epoch of the validation set for each neuron). That means we need to find a proxy/approximation of this metric.

The challenge of this idea are:

- Finding a good approxmation of the metric
- Finding a stable order (does not change too quickly so that neurons goes randomly from 0 to 1 activation all screw the downstream weights)
- Make sure that the new neurons integrate well in the the network. Otherwise they would always stay out the network and it could not reach it's optimal size

### Let the gradient descent "learn" the order

The idea behind this solution is. *If we don't know which neurons are important, let's ask the network*. In the previous method (with the sigmoid multiplicative factor). We were generating a vector with zeros and ones (ones on the left and zeros on the right), 1 for a neuron that is alive, 0 for dead. We only had one parameter, *The number of neurons*. Here instead we would have one additional parameter for each neuron, *dead or alive*. And we would optimize them. 

Of course if we let things this way, no neuron will be killed. But it makes sense. If we don't penalize the network he will use as many neurons as possible. We want to make sure that the useless ones are killed. The key idea is to apply regularization on this dead/alive vector. and to make sure that some neurons are killed we want some sparity inducing regularlization (L1 for example).

In this situation we would not have any order in fact. Each neuron would be independent.

The challenges of this potential solution are:

- **How to know when to add new neurons ?**. Out of the box this solution only allows us to know which neurons to kill (when they reach 0) but, we don't know when/how to add more neurons to the network.
- **What size should we start with**. Since we can only shrink the network. What is a good size to start with. If we undershoot we will not have the best network and if we overshoot we will spend significant time at the beginining of the training process working on useless neurons that will be killed eventually.

## Conclusion

We have two strong potential ideas to solve our main problem. Fow now we will go with the second one because it does not involve the long process of finding a metric (try multiple, compare with the perfect baseline...).

# Implement the first version of a *Sparsifier Network*

## Description

To test the potential solution we decided to go for we need to have a first draft of an implementation and run tests on it.


|Start Date|End Date  |
|----------|----------|
|2017-11-10|2017-11-10|

## Delivrables

- [x] Pytorch Implementation of a simple one layer sparsifier network
  - Implemented in `models/MNIST_1h_sparsifier.py`
- [x] Plots 
- [ ] Interepretation
- [ ] Conclusion

## Interpretation

The experiment we performed on the implementation of the sparsifier network is:

- We picked models with a starting size of 500
- Trained them 60 epochs with different factors on the L1 regularization term

As we can see from the results on `MNIST`, the final size increase as the penalty decrease. Which makes sense. The models train very well. (They rapidly reach 100% testing accuracy). And the two last ones (roughly at 100 and 200 neurons) generalize better than the static network (the one with no penlaty).

![MNIST - simple sparse network training results](/plots/MNIST_1h_stats_strict_sparsifier.png?raw=true "MNIST - simple sparse network training results")

On `FashionMNIST` we can observe similary good results except that the accuracies are lower because the problem is harder. We also see that the netowrk with a penalty of 10^-4 is better at **fitting and generalizing** (The difference might be statistically not significant though).

![FashionMNIST - simple sparse network training results](/plots/FashionMNIST_1h_stats_strict_sparsifier.png?raw=true "FashionMNIST - simple sparse network training results")

Without any doubt, the most interesing observation we can make is that for the first time since this project started we can clearly see that `FashionMNIST` is more complex than `MNIST`. For a given penalty the `FashionMNIST` size is always higher than `MNIST`. This is a very good sign and a clear advantage for this technique

As we can see from these two plots (`MNIST` and then `FashionMNIST`), the training process is smooth except at epoch 2. The reason for that is that this is the moment when some neuron liveness starts getting set to 0 and are therefore removed from the network. Some neurons downstream needed their values so they have to overcome the fact that they were killed

![MNIST - sparsifier network training process](/plots/MNIST_1h_training_strict_sparsifier.png?raw=true "MNIST - sparsifier network training process")

![FashionMNIST - sparsifier network training process](/plots/FashionMNIST_1h_training_strict_sparsifier.png?raw=true "FashionMNIST - sparsifier network training process")


## Conclusion

Even if this solution seems very promising, It only has the ability of shrinking models. It won't help us finding the optimal size for a network.

Also one question remains unanswered: *Does the starting size influence the size after convergence ?* 




# Evaluate Inference time influence of multiple neurons orderings

## Description

In the previous model we do not really see very good accuracy. I think the main issue is that the limit between useful and useless neurons is completely arbirary. If the interesting neurons are on the "right", then they should not be discarded. If we could find a way of order them by importance and train the limit (x_0) between important and less important neurons. A perfect ordering would be "the decreasing impact on the loss if this neuron would be set to zero". We will try to find if this quantity is easy to evaluate, or estimate. If not we will try to find a proxy for this metric.

|Start Date|End Date  |
|----------|----------|
|2017-10-25|          |

## Delivrables

- [ ] Try to find if there is a way to compute the increase in loss if a given neuron would be set to zero
- [ ] Try to find proxies for this ordering
- [ ] Generate plots
- [ ] Interpret
- [ ] Conclude

# Evaluate the stability of these orderings at training time

## Description

Neural networks expect smooth variations during trainig. Reordering neurons will change their scaler. If the reorder is more than one or two spots wide then the activation might go from 0 to 1 in a single batch. It might take a while before the rest of the network adapt from this change. A good ordering of the neurons would make sure that the number of swaps is low and that neurons move a small distance durint trianing

|Start Date|End Date  |
|----------|----------|
|          |          |

## Delivrables

- [ ] Implement a benchmark in pytorch that compare the stability for these metrics
- [ ] Generate plots
- [ ] Interpret
- [ ] Conclude

