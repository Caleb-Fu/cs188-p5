import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.
        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.
        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        Returns: 1 or -1
        """
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        still_learning = True
        loss_diff = 999
        last_loss = None
        while still_learning:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                still_learning = False
                if last_loss:
                    loss_diff = abs(nn.as_scalar(loss)-nn.as_scalar(last_loss))
                last_loss = loss
                if loss_diff > 0.00001:
                    still_learning = True
                    grads = nn.gradients(loss, [self.first_weights, self.fb, self.second_weights, self.sb, self.tw, self.tb])
                    self.first_weights.update(grads[0], self.learning_rate)
                    self.fb.update(grads[1], self.learning_rate)
                    self.second_weights.update(grads[2], self.learning_rate)
                    self.sb.update(grads[3], self.learning_rate)
                    self.tw.update(grads[4], self.learning_rate)
                    self.tb.update(grads[5], self.learning_rate)

                    

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 10
        self.learning_rate = -0.001
        self.first_w = nn.Parameter(1, 15)
        self.first_b = nn.Parameter(1,15)
        self.second_w = nn.Parameter(15, 10)
        self.second_b = nn.Parameter(1,10)
        self.third_w = nn.Parameter(10,1)
        self.third_b = nn.Parameter(1,1)
        
    def run(self, x):
        """
        Runs the model for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        first = nn.ReLU(nn.AddBias(nn.Linear(x, self.first_w), self.first_b))
        second = nn.ReLU(nn.AddBias(nn.Linear(first, self.second_w), self.second_b))
        third = nn.AddBias(nn.Linear(second, self.third_w), self.third_b)
        return third

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) > 0.015:
                    grads = nn.gradients(loss, [self.first_w, self.first_b, self.second_w, self.second_b, self.third_w, self.third_b])
                    self.first_w.update(grads[0], self.learning_rate)
                    self.first_b.update(grads[1], self.learning_rate)
                    self.second_w.update(grads[2], self.learning_rate)
                    self.second_b.update(grads[3], self.learning_rate)
                    self.third_w.update(grads[4], self.learning_rate)
                    self.third_b.update(grads[5], self.learning_rate)
                else:
                    return



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.
    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.
    The goal is to sort each digit into one of 10 classes (number 0 through 9).
    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.learning_rate = -0.005
        self.batch_size = 1
        self.first_w = nn.Parameter(784, 100)
        self.first_b = nn.Parameter(1,100)
        self.second_w = nn.Parameter(100, 20)
        self.second_b = nn.Parameter(1, 20)
        self.third_w = nn.Parameter(20,10)
        self.third_b = nn.Parameter(1,10)
        
    def run(self, x):
        """
        Runs the model for a batch of examples.
        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.
        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        first = nn.ReLU(nn.AddBias(nn.Linear(x, self.first_w), self.first_b))
        second = nn.ReLU(nn.AddBias(nn.Linear(first, self.second_w), self.second_b))
        third = nn.AddBias(nn.Linear(second, self.third_w), self.third_b)
        return third

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    
    def train(self, dataset):
        """
        Trains the model.
        """
        
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                if dataset.get_validation_accuracy() < 0.97:
                    grads = nn.gradients(loss, [self.first_w, self.first_b, self.second_w, self.second_b, self.third_w, self.third_b])
                    self.first_w.update(grads[0], self.learning_rate)
                    self.first_b.update(grads[1], self.learning_rate)
                    self.second_w.update(grads[2], self.learning_rate)
                    self.second_b.update(grads[3], self.learning_rate)
                    self.third_w.update(grads[4], self.learning_rate)
                    self.third_b.update(grads[5], self.learning_rate)
                else:
                    return

        
