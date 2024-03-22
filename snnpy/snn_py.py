import atexit, neuralnetwork as nn, _lib_util as lu, ctypes

def _cleanup_nn(neural_network: nn.NeuralNetwork):
    lib = lu._get_lib()
    lib.freeNeuralNetwork(neural_network.c_nn_ptr)
    
def create_neural_network(nr_inputs: int, nr_layers: int, neurons_p_layer: int, nr_outputs: int) -> nn.NeuralNetwork | None:
    '''
        Creates a neural network with the specified number of inputs, hidden layers, neurons per layer and outputs
    '''
    if not all(isinstance(arg, int) for arg in [nr_inputs, nr_layers, neurons_p_layer, nr_outputs]):
        raise TypeError("create_neural_network arguments must be integers")
    lib = lu._get_lib()
    neural_network: nn.NeuralNetwork = nn.NeuralNetwork(nr_inputs, nr_layers, neurons_p_layer, nr_outputs)
    lib.initNeuralNetwork(neural_network.c_nn_ptr, nr_inputs, nr_layers, neurons_p_layer, nr_outputs)
    atexit.register(_cleanup_nn, neural_network)
    return neural_network

  
        


    

