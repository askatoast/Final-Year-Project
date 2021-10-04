# Final-Year-Project
The idea of CartPole is that there is a pole standing up on top of a cart. The goal is to balance this pole by wiggling/moving the cart from side to side to keep the pole balanced upright.

The environment is deemed successful if we can balance for 200 frames, and failure is deemed when the pole is more than 15 degrees from fully vertical.

Every frame that we go with the pole "balanced" (less than 15 degrees from vertical), our "score" gets +1, and our target is a score of 200(we have used 500).

Now, how do we do this? There are endless ways, some very complex, and some very specific. I'd like to solve this very generally, and in a way that we could easily apply this same solution to a wide variety of problems.

This will also give me the ability to illustrate a very interesting property of neural networks. If you've ever taken a statistics course, you might be familiar with the scenario where you can have various signals, which have some degree of predictive power, and combine them for something with more predictive power than the sum of the parts.

Neural networks are fully capable of doing this on their own entirely.

To illustrate this, we're going to start by creating an agent that, when in this cartpole environment, it just randomly chooses actions (left and right). Recall that our goal is to get a score of 200, but we'll go ahead and use any scenario where we've scored above 50 to learn from.

1) initial_population function takes a random action and generates the output based on that random action this is done to train the initial data which finally returns the training_data
 

After this we created a neural network 

def nn_model(input_size):
first we create a input layer so it takes a input_data the shape of the data which basically contains of a input size and the number which is None at the start

now we create out fully connected layers which takes the input whihc the network and has 128/256/512 nodes at the layer and activation as rectified linear

Rectified Linear(The rectified linear activation function overcomes the vanishing gradient problem, allowing models to learn faster and perform better. The rectified linear activation is the default activation when developing multilayer Perceptron and convolutional neural networks.)

Then for every fully connected network layer (here we have 5) we give a dropout rate in this case it is 0.2 which is 20% of the input is dropped

after this we make an output layer which take the network and gives 2 output (since we only move left and right here) and the activation here will be softmax

Now we do regression  against the network and the optimizer adam is used with the learning rate defined at the beginning of the code and the for calculating loss we used categorical_crossentropy 

and we return the model


Now we train the model since the nn model is not trained using out traing_data generted from the initial population 
Here we take 2 variables X and Y
X stores the observations of the training data and Y stores the moves an index at a time 
we reshaped the observations before storing it to X

Now we do a model.fit which basically trains the model here it takes X as input and Y as the corresponding output 
we have tested it for 3 epochs
Epoch(An epoch means training the neural network with all the training data for one cycle. In an epoch, we use all of the data exactly once.)

and we return the model 

Now the model is trained 

So now we just play the game using that particular model 




