from __future__ import annotations
import ctypes

# defining function types from the shared library
DBLA_DBLR = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
DBLA_DBLR_DBLR = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
DBLPA_DBLPA_INTA_DBLR = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int)

class Neuron(ctypes.Structure):
    '''
        Python copy representing the Neuron struct in the shared library
    '''
    def __init__(self):
        self._fields_ = [
            ("A", ctypes.c_double),
            ("Z", ctypes.c_double),
            ("bias", ctypes.c_double),
            ("connections", ctypes.c_int),
            ("weights", ctypes.POINTER(ctypes.c_double)),
            ("activationFunctions", ctypes.POINTER(DBLA_DBLR)),
            ("connectedNeurons", ctypes.POINTER(ctypes.POINTER(Neuron))),
        ]        

NA_INTA_DBLR = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.POINTER(Neuron)), ctypes.c_int)

class NeuralNetwork(ctypes.Structure):
    '''
        Python copy representing the NeuralNetwork struct in the shared library
    '''
    _fields_ = [
        ("nrOfInputNeurons", ctypes.c_int),
        ("nrOfHiddenNeurons", ctypes.c_int),
        ("nrOfOutputNeurons", ctypes.c_int),
        ("nrOfNeurons", ctypes.c_int),
        ("nrOfHiddenLayers", ctypes.c_int),
        ("neuronsPerLayer", ctypes.c_int),
        ("costFunction", DBLPA_DBLPA_INTA_DBLR),
        ("regularizationDerivative", DBLA_DBLR_DBLR),
        ("regularization", NA_INTA_DBLR),
        ("regularizationDerivative", DBLA_DBLR),
        ("inputLayerActivationFunctions", ctypes.POINTER(DBLA_DBLR)),
        ("hiddenLayerActivationFunctions", ctypes.POINTER(DBLA_DBLR)),
        ("outputLayerActivationFunctions", ctypes.POINTER(DBLA_DBLR)),
        ("neurons", ctypes.POINTER(ctypes.POINTER(Neuron))),
        ("outputLayer", ctypes.POINTER(ctypes.POINTER(Neuron)))
    ]
        
class NeuronGradient(ctypes.Structure):
    '''
        Python copy representing the NeuronGradient struct in the shared library
    '''
    _fields_ = [
        ("nrOfWeights", ctypes.c_int),
        ("weightGradient", ctypes.POINTER(ctypes.c_double)),
        ("biasGradient", ctypes.c_double),
    ]

class GradientVector(ctypes.Structure):
    '''
        Python copy representing the GradientVector struct in the shared library
    '''
    _fields_ = [
        ("nrOfNeurons", ctypes.c_int),
        ("gradients", ctypes.POINTER(ctypes.POINTER(NeuronGradient)))
    ]

class GradientBatch(ctypes.Structure):
    '''
        Python copy representing the Batch struct in the shared library
    '''
    _fields_ = [
        ("batchSize", ctypes.c_int),
        ("gradientVectors", ctypes.POINTER(ctypes.POINTER(GradientVector)))
    ]

class PyNeuralNetwork:
    '''
        A Python object wrapper for a C NeuralNetwork struct from the shared library
    '''
    def __init__(self, nr_of_inputs: int, nr_of_layers: int, neurons_per_layer: int, nr_of_outputs: int):
        temp_struct = NeuralNetwork()
        self.c_nn_ptr = ctypes.pointer(temp_struct)
        self.nr_of_inputs = nr_of_inputs
        self.nr_of_layers = nr_of_layers
        self.neurons_per_layer = neurons_per_layer
        self.nr_of_outputs = nr_of_outputs
        


        
        