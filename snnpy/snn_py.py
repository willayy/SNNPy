import atexit, ctypes
from __init__ import LIB
from neuralnetwork import NeuralNetwork

def _cleanup_nn(neural_network: nn.NeuralNetwork):
    '''
        Cleans up the memory allocated for a neural network at script exit
    '''
    LIB.freeNeuralNetwork(neural_network.c_nn_ptr)
    
def _get_activation_function(name: str) -> ctypes._FuncPointer:
    '''
        Returns the activation function with the specified name
    '''

    try:
        case_ac_f = {
            "sigmoid": LIB.sigmoid,
            "relu": LIB.rectifiedLinearUnit,
            "tanh": LIB.hyperbolicTangent,
            "linear": LIB.linear
        }
    except Exception as e:
        print(e)
        raise ValueError(f"Activation function {name} not found")

    return ctypes.cast(case_ac_f[name], ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double))

def _get_activation_function_derivative(name: str) -> ctypes._FuncPointer:
    '''
        Returns the derivative of the activation function with the specified name
    '''

    try:
        case_ac_f_derivative = {
            "sigmoid": LIB.sigmoidDerivative,
            "relu": LIB.rectifiedLinearUnitDerivative,
            "tanh": LIB.hyperbolicTangentDerivative,
            "linear": LIB.linearDerivative
        }
    except Exception as e:
        print(e)
        raise ValueError(f"Activation function {name} not found")
    
    return ctypes.cast(case_ac_f_derivative[name], ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double))

# Initialization methods

def initialize_rng(seed: int) -> None:
    '''
        Assures that the random number generator is initialized
    '''
    arg = ctypes.c_int(seed)
    LIB.initRNG(arg)

def get_rng_seed() -> int:
    '''
        Returns the seed of the random number generator
    '''
    return int(LIB.getSeed())

def create_neural_network(nr_inputs: int, nr_hidden_layers: int, neurons_p_layer: int, nr_outputs: int) -> NeuralNetwork | None:
    '''
        Creates a neural network with the specified number of inputs, hidden layers, neurons per layer and outputs
    '''
    if not all(isinstance(arg, int) for arg in [nr_inputs, nr_hidden_layers, neurons_p_layer, nr_outputs]):
        raise TypeError("create_neural_network arguments must be integers")
    neural_network: NeuralNetwork = NeuralNetwork(nr_inputs, nr_hidden_layers, neurons_p_layer, nr_outputs)
    LIB.initNeuralNetwork(neural_network.c_nn_ptr, nr_inputs, nr_hidden_layers, neurons_p_layer, nr_outputs)
    atexit.register(_cleanup_nn, neural_network)
    return neural_network

def set_input_activation_function(neural_network: NeuralNetwork, activation_function: str) -> None:
    '''
        Sets the activation function for the input layer
        ### Args:
            * #### "relu": Rectified Linear Unit
            * #### "sigmoid": Sigmoid
            * #### "tanh": Hyperbolic Tangent
            * #### "linear": Linear
    '''
    activation_function = _get_activation_function(activation_function)
    activation_function_derivative = _get_activation_function_derivative(activation_function)
    LIB.setInputLayerActivationFunction(neural_network.c_nn_ptr, activation_function, activation_function_derivative)

def set_hidden_layer_activation_function(neural_network: NeuralNetwork, activation_function: str) -> None:
    '''
        Sets the activation function for the hidden layers
        ### Args:
            * #### "relu": Rectified Linear Unit
            * #### "sigmoid": Sigmoid
            * #### "tanh": Hyperbolic Tangent
            * #### "linear": Linear
    '''
    activation_function = _get_activation_function(activation_function)
    activation_function_derivative = _get_activation_function_derivative(activation_function)
    LIB.setHiddenLayerActivationFunction(neural_network.c_nn_ptr, activation_function, activation_function_derivative)
  
def set_output_layer_activation_function(neural_network: NeuralNetwork, activation_function: str) -> None:
    '''
        Sets the activation function for the output layer
        ### Args:
            * #### "relu": Rectified Linear Unit
            * #### "sigmoid": Sigmoid
            * #### "tanh": Hyperbolic Tangent
            * #### "linear": Linear
    '''
    activation_function = _get_activation_function(activation_function)
    activation_function_derivative = _get_activation_function_derivative(activation_function)
    LIB.setOutputLayerActivationFunction(neural_network.c_nn_ptr, activation_function, activation_function_derivative)

def init_weights_xavier_normal(neural_network: NeuralNetwork) -> None:
    LIB.initWeightsXavierNormal(neural_network.c_nn_ptr)

def init_weights_xavier_uniform(neural_network: NeuralNetwork) -> None:
    LIB.initWeightsXavierUniform(neural_network.c_nn_ptr)

def init_weights_random_uniform(neural_network: NeuralNetwork, min_w: float, max_w: float) -> None:
    min_w_arg = ctypes.c_double(min_w)
    max_w_arg = ctypes.c_double(max_w)
    LIB.initWeightsRandomUniform(neural_network.c_nn_ptr, min_w_arg, max_w_arg)

def init_biases_random_uniform(neural_network: NeuralNetwork, min_b: float, max_b: float) -> None:
    min_b_arg = ctypes.c_double(min_b)
    max_b_arg = ctypes.c_double(max_b)
    LIB.initBiasesRandomUniform(neural_network.c_nn_ptr, min_b_arg, max_b_arg)

def init_biases_constant(neural_network: NeuralNetwork, bias_v: float) -> None:
    bias_arg = ctypes.c_double(bias_v)
    LIB.initBiasesConstant(neural_network.c_nn_ptr, bias_arg)

# Training methods
    
def train_neural_network(neural_network: NeuralNetwork,
                         inputs: list[list[float]],
                         desired_outputs: list[list[float]],
                         batch_size: int,
                         amount_epochs: int,
                         learing_rate_w: float,
                         learning_rate_b: float,
                         lambda_reg: float = 0,
                         l1_reg: bool = False,
                         l2_reg: bool = False,
                         verbose: bool = True) -> None:
    '''
        Trains the neural network with the specified inputs and desired outputs. The
        training is done by dividing up the inputs in to the specified batch size, shuffling them, 
        calculating the gradients, averaging them and updating the weights and biases. \n
        ## Args:
            * #### neural_network: The neural network to train
            * #### inputs: The input data
            * #### desired_outputs: The desired output data
            * #### batch_size: The size of the batches to use
            * #### amount_epochs: The amount of epochs to train for
            * #### learing_rate_w: The learning rate for the weights
            * #### learning_rate_b: The learning rate for the biases
            * #### lambda_reg: The regularization parameter
            * #### l1_reg: Whether to use L1 regularization
            * #### l2_reg: Whether to use L2 regularization
            * #### verbose: Whether to print the training progress  
    '''

    pass