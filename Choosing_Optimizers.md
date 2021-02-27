## Notes to select optimizers for your Deep Learning models based on ![[1]](https://ruder.io/optimizing-gradient-descent/index.html)

1) Model is a function approximated by a neural network after training.
	- eg: a simple model is y = Wx + b, where W and b 			are model parameters.

2) To determine how good a fit is, loss function is used.
	- eg: squared error loss function
		L(y, t) = (1/2) * (y – t) **2

3) When the model and loss function are combined we get an optimization problem. 

4) Objective function is a loss function in which the model is plugged in i.e., it is parameterized by model parameters.

5) Optimization involves minimizing an objective/cost function with respect to model parameters.

6) Gradient descent is used to optimize neural networks. 

7) A gradient is obtained by differentiating(calculus) the objective function with respect to the parameters.

8) Minimization means to update the parameters in the opposite direction of the gradient.

9) Learning rate is the size of the step taken to reach a global/local minima.

### SGD(Stochastic Gradient Descent):
1) SGD iteratively updates the gradients by calculating it for one example at a time.

2) It is fast and can be used for online learning.

3) In mini-batch gradient descent, instead of iterating over each examples, the gradient is calculated on a batch size of n.
 

4) Frequent updates with high variance result in high fluctuations.

5) Fluctuations result in jumping from one local minima to the other complicating convergence.

6) But, it is shown that when the learning rate is decreased slowly, it converges to local or global minima.

7) Usually when SGD is mentioned, it means SGD using mini-batches.

### Challenges posed by SGD:
1) It is difficult to choose a proper learning rate

2) Learning rate can be reduced according to a pre-defined schedule or depending upon when the change in the objective between epochs falls below a threshold. These schedules and thresholds have to be defined in advance and wont adapt to dataset’s characteristics.

3) The same learning rate is applied to all parameters.

4) Escaping from suboptimal local minima and saddle points traps.

### Momentum:
1) Accelerates SGD in the appropriate direction and also reduces oscillations.

### NAG(Nesterov accelerated gradient):
1) NAG is is better than Momentum.

2) The anticipatory NAG update prevents us from going too fast and results in increased responsiveness.

3) Updates are adapted to the slope of the error function and results in more speed than SGD.

### Adagrad:
1) Has per parameter learning rate and uses low learning rates for high frequency features and high learning rates for low frequency features.

2) Suitable for dealing with sparse data.

3) No manual tuning of learning rates is required. 

4) Default learning rate is 0.01

5) Learning rate of this algorithm shrinks.

Note: if the learning rate is infinitesimally small, the algorithm cannot learn.

### Adadelta:
1) This is an extension of Adagrad that seeks to reduce its aggressive, diminishing learning rate. 

2) No need to set a default learning rate.

### RMSprop:
1) Solves Adagrad's radically diminishing learning rates.

2) Suggested default values: Momentum, γ = 0.9, learning rate, η = 0.001.

### Adam(Adaptive Moment Estimation):
1) Adam is another method that computes adaptive learning rates for each parameter.

2) Adam can be viewed as a combination of RMSprop and momentum.

3) Default values are 0.9 for β1, 0.999 for β2, and 10−8 for ϵ. 

### Adamax:
1) Updates are more stable.

2) Default values are η=0.002, β1=0.9, and β2=0.999.

### Nadam(Nesterov-accelerated Adaptive Moment Estimation):
1) Nadam   combines Adam and NAG.

### AMSGrad:
1) Adaptive learning rate methods in some cases are outperformed by SGD with momentum.

2) The solution of Adaptive learning rate methods has the following disadvantages:
    • Diminishes the influence of large and informative gradients which leads to poor convergence. 
    • Results in short-term memory of the gradients which becomes an obstacle in other scenarios.

3) Because of the above reasons, the following algorithms have poor generalization behaviour: Adadelta, RMSprop, Adam, AdaMax, and Nadam

4) AMSGrad results in a non-increasing step size, which results in good generalization behavior.

References:

1.![https://ruder.io/optimizing-gradient-descent/index.html](https://ruder.io/optimizing-gradient-descent/index.html)

2.![https://d2l.ai/d2l-en.pdf](https://d2l.ai/d2l-en.pdf)

3.Lecture notes from : ![https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/](https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/)
