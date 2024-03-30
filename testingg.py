from snnpy import snn_py

# Define the dataset inputs
inputs: list[list[float]] = [[0,0,0,0],
                             [0,0,0,1],
                             [0,0,1,0],
                             [0,0,1,1],
                             [0,1,0,0],
                             [0,1,0,1],
                             [0,1,1,0],
                             [0,1,1,1],
                             [1,0,0,0],
                             [1,0,0,1],
                             [1,0,1,0],
                             [1,0,1,1],
                             [1,1,0,0],
                             [1,1,0,1],
                             [1,1,1,0],
                             [1,1,1,1]]

# Define the dataset labels
outputs: list[list[float]] = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]

snn_py.set_rng_seed(0) # Set the seed for the random number generator

neural_network = snn_py.create_neural_network(4,1,4,16) # Create a neural network model

snn_py.set_activation_functions(neural_network, 'linear', 'relu', 'sigmoid') # Set the activation functions for each layer (layers* in the case of hidden layers)
snn_py.set_cost_function(neural_network, 'cross_entropy') # Set the cost function for the model
snn_py.set_regularization(neural_network, 'no_reg') # Set the regularization for the model
snn_py.init_weights_xavier_normal(neural_network) # Initialize the weights of the model
snn_py.init_biases_constant(neural_network, 0.1) # Initialize the biases of the model

# Train the model on a batch from the dataset
snn_py.train_neural_network(neural_network, inputs, outputs, 16, 10000, 0.08, 0.01)