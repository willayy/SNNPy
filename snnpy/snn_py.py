import atexit, ctypes
from snnpy import c_lib
from snnpy.neuralnetwork import NeuralNetwork

def _cleanup_nn(neural_network: NeuralNetwork):
    '''
        Cleans up the memory allocated for a neural network at script exit
    '''
    c_lib.freeNeuralNetwork(neural_network.c_nn_ptr)

def _get_activation_function(name: str) -> ctypes.pointer:
    '''
        Returns the activation function with the specified name
    '''

    func_dict = {
        "linear": c_lib.linear,
        "sigmoid": c_lib.sigmoid,
        "relu": c_lib.rectifiedLinearUnit,
        "tanh": c_lib.hyperbolicTangent
    }

    func = func_dict.get(name)
    
    if func is None:
        raise ValueError(f"Activation function {name} not found")

    return func

def _get_activation_function_derivative(name: str) -> ctypes.pointer:
    '''
        Returns the derivative of the activation function with the specified name
    '''

    func_dict = {
        "linear": c_lib.linearDerivative,
        "sigmoid": c_lib.sigmoidDerivative,
        "relu": c_lib.rectifiedLinearUnitDerivative,
        "tanh": c_lib.hyperbolicTangentDerivative
    }

    func = func_dict.get(name)

    if func is None:
        raise ValueError(f"Activation function {name} not found")
    
    return func

def _get_cost_function(name: str) -> ctypes.pointer:
    '''
        Returns the cost function with the specified name
    '''

    case_cost_f = {
        "mse": c_lib.sqrCostFunction,
        "cross_entropy": c_lib.crossEntropyCostFunction
    }
    func = case_cost_f[name]
    if func is None:
        raise ValueError(f"Cost function {name} not found")

    return func

def _get_cost_function_derivative(name: str) -> ctypes.pointer:
    '''
        Returns the derivative of the cost function with the specified name
    '''

    case_cost_f_derivative = {
        "mse": c_lib.sqrCostFunctionDerivative,
        "cross_entropy": c_lib.crossEntropyCostFunctionDerivative
    }
    func = case_cost_f_derivative[name]
    if func is None:
        raise ValueError(f"Cost function {name} not found")
    
    return func

def _get_regularization(name: str) -> ctypes.pointer:
    '''
        Returns the regularization function with the specified name
    '''

    case_reg_f = {
        "l1": c_lib.l1Regularization,
        "l2": c_lib.l2Regularization,
        "no_reg": c_lib.noRegularization
    }
    func = case_reg_f[name]
    if func is None:
        raise ValueError(f"Regularization function {name} not found")

    return func

def _get_regularization_derivative(name: str) -> ctypes.pointer:
    '''
        Returns the derivative of the regularization function with the specified name
    '''

    case_reg_f_derivative = {
        "l1": c_lib.l1RegularizationDerivative,
        "l2": c_lib.l2RegularizationDerivative,
        "no_reg": c_lib.noRegularizationDerivative
    }
    func = case_reg_f_derivative[name]
    if func is None:
        raise ValueError(f"Regularization function {name} not found")
    
    return func

def _python_2d_list_to_c_array(py_list: list[list[float]]) -> ctypes.pointer:
    '''
        Converts a 2D python list to a 2D C array
    '''
    c_array = (ctypes.c_double * len(py_list[0]) * len(py_list))()
    for i in range(len(py_list)):
        for j in range(len(py_list[0])):
            c_array[i][j] = ctypes.c_double(py_list[i][j])
    return c_array

# Initialization methods

def set_rng_seed(seed: int) -> None:
    '''
        Assures that the random number generator is initialized
    '''
    arg = ctypes.c_int(seed)
    c_lib.setRngSeed(arg)

def get_rng_seed() -> int:
    '''
        Returns the seed of the random number generator
    '''
    return int(c_lib.getSeed())

def create_neural_network(nr_inputs: int, nr_hidden_layers: int, neurons_p_layer: int, nr_outputs: int) -> NeuralNetwork | None:
    '''
        Creates a neural network with the specified number of inputs, hidden layers, neurons per layer and outputs
    '''
    if not all(isinstance(arg, int) for arg in [nr_inputs, nr_hidden_layers, neurons_p_layer, nr_outputs]):
        raise TypeError("create_neural_network arguments must be integers")
    neural_network: NeuralNetwork = NeuralNetwork(nr_inputs, nr_hidden_layers, neurons_p_layer, nr_outputs)
    c_lib.initNeuralNetwork(neural_network.c_nn_ptr, nr_inputs, nr_hidden_layers, neurons_p_layer, nr_outputs)
    atexit.register(_cleanup_nn, neural_network)
    return neural_network

def set_activation_functions(neural_network: NeuralNetwork, input_layer: str, hidden_layer: str, output_layer: str) -> None:
    '''
        Sets the activation function for the layers of the neural network, all hidden layers will have the same activation function
        ### Args:
            * #### "relu": Rectified Linear Unit
            * #### "sigmoid": Sigmoid
            * #### "tanh": Hyperbolic Tangent
            * #### "linear": Linear
    '''
    input_layer_f = _get_activation_function(input_layer)
    input_layer_derivative_f = _get_activation_function_derivative(input_layer)
    hidden_layer_f = _get_activation_function(hidden_layer)
    hidden_layer_derivative_f = _get_activation_function_derivative(hidden_layer)
    output_layer_f = _get_activation_function(output_layer)
    output_layer_derivative_f = _get_activation_function_derivative(output_layer)

    c_lib.setInputLayerActivationFunction(neural_network.c_nn_ptr, input_layer_f, input_layer_derivative_f)
    c_lib.setHiddenLayerActivationFunction(neural_network.c_nn_ptr, hidden_layer_f, hidden_layer_derivative_f)
    c_lib.setOutputLayerActivationFunction(neural_network.c_nn_ptr, output_layer_f, output_layer_derivative_f)

def set_cost_function(neural_network: NeuralNetwork, cost_function: str) -> None:
    cost_function_f = _get_cost_function(cost_function)
    cost_function_derivative_f = _get_cost_function_derivative(cost_function)
    c_lib.setCostFunction(neural_network.c_nn_ptr, cost_function_f, cost_function_derivative_f)

def set_regularization(neural_network: NeuralNetwork, regularization: str) -> None:
    regularization_f = _get_regularization(regularization)
    regularization_derivative_f = _get_regularization_derivative(regularization)
    c_lib.setRegularization(neural_network.c_nn_ptr, regularization_f, regularization_derivative_f)

def init_weights_xavier_normal(neural_network: NeuralNetwork) -> None:
    c_lib.initWeightsXavierNormal(neural_network.c_nn_ptr)

def init_weights_xavier_uniform(neural_network: NeuralNetwork) -> None:
    c_lib.initWeightsXavierUniform(neural_network.c_nn_ptr)

def init_weights_random_uniform(neural_network: NeuralNetwork, min_w: float, max_w: float) -> None:
    min_w_arg = ctypes.c_double(min_w)
    max_w_arg = ctypes.c_double(max_w)
    c_lib.initWeightsRandomUniform(neural_network.c_nn_ptr, min_w_arg, max_w_arg)

def init_biases_random_uniform(neural_network: NeuralNetwork, min_b: float, max_b: float) -> None:
    min_b_arg = ctypes.c_double(min_b)
    max_b_arg = ctypes.c_double(max_b)
    c_lib.initBiasesRandomUniform(neural_network.c_nn_ptr, min_b_arg, max_b_arg)

def init_biases_constant(neural_network: NeuralNetwork, bias_v: float) -> None:
    bias_arg = ctypes.c_double(bias_v)
    c_lib.initBiasesConstant(neural_network.c_nn_ptr, bias_arg)

# Training methods
    
def train_neural_network(neural_network: NeuralNetwork,
                         inputs: list[list[float]],
                         labels: list[list[float]],
                         batch_size: int,
                         amount_epochs: int,
                         learing_rate_w: float,
                         learning_rate_b: float,
                         lambda_reg: float = 0,
                         verbose: bool = True) -> None:
    '''
        Trains the neural network with the specified inputs and desired outputs. The
        training is done by dividing up the inputs in to the specified batch size, shuffling them, 
        calculating the gradients, averaging them and updating the weights and biases. \n
        ## Args:
            * #### neural_network: 
            The neural network to train 
            * #### inputs: 
            The input data
            * #### desired_outputs:
            The desired output data
            * #### batch_size:
            The size of the batches to use
            * #### amount_epochs:
            The amount of epochs to train for
            * #### learing_rate_w:
            The learning rate for the weights
            * #### learning_rate_b:
            The learning rate for the biases
            * #### lambda_reg:
            The regularization parameter
            * #### l1_reg:
            Whether to use L1 regularization
            * #### l2_reg:
            Whether to use L2 regularization
            * #### verbose:
            Whether to print the training progress  
    '''
    if length := len(inputs) != len(labels):
        raise ValueError(f"inputs and labels must have the same length, got {length} and {len(labels)}")
    if not all(len(inputs[i]) == neural_network.nr_of_inputs and len(labels[i]) for i in range(length)):
        raise ValueError("inputs and labels must have the same length for each element")

    double_ptr_inputs = _python_2d_list_to_c_array(inputs)
    double_ptr_labels = _python_2d_list_to_c_array(labels)
    learing_rate_w = ctypes.c_double(learing_rate_w)
    learning_rate_b = ctypes.c_double(learning_rate_b)
    lambda_reg = ctypes.c_double(lambda_reg)
    batch_size = ctypes.c_int(batch_size)
    amount_epochs = ctypes.c_int(amount_epochs)

    verbose = ctypes.c_int(1) if verbose else ctypes.c_int(0)

    c_lib.trainNeuralNetworkOnBatch(neural_network.c_nn_ptr, double_ptr_inputs, double_ptr_labels, amount_epochs, 
                                    batch_size, learing_rate_w, learning_rate_b, lambda_reg, verbose)
    