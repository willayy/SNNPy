from __future__ import annotations
import ctypes


class CNeuron(ctypes.Structure):
    '''
        Python copy representing the Neuron struct in the shared library
    '''
    def __init__(self):
        self._fields_ = [
            ("A", ctypes.c_double),
            ("Z", ctypes.c_double),
            ("bias", ctypes.c_double),
            ("connecttions", ctypes.c_int),
            ("weights", ctypes.POINTER(ctypes.c_double)),
            ("activationFunction", ctypes.POINTER((ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)))),
            ("connectedNeurons", ctypes.POINTER(ctypes.POINTER(CNeuron))),
        ]        


class CNeuralNetwork(ctypes.Structure):
    '''
        Python copy representing the NeuralNetwork struct in the shared library
    '''
    def __init__(self):
        self._fields_ = [
            ("nrOfInputNeurons", ctypes.c_int),
            ("nrOfHiddenNeurons", ctypes.c_int),
            ("nrOfOutputNeurons", ctypes.c_int),
            ("nrOfNeurons", ctypes.c_int),
            ("nrOfHiddenLayers", ctypes.c_int),
            ("neuronsPerLayer", ctypes.c_int),
            ("inputLayerActivationFunction", ctypes.POINTER((ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)))),
            ("hiddenLayerActivationFunction", ctypes.POINTER((ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)))),
            ("outputLayerActivationFunction", ctypes.POINTER((ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)))),
            ("neurons", ctypes.POINTER(ctypes.POINTER(CNeuron))),
            ("outputLayer", ctypes.POINTER(ctypes.POINTER(CNeuron)))
        ]

class CNeuronGradient(ctypes.Structure):
    '''
        Python copy representing the NeuronGradient struct in the shared library
    '''
    def __init__(self):
        self._fields_ = [
            ("nrOfWeights", ctypes.c_int),
            ("weightGradient", ctypes.POINTER(ctypes.c_double)),
            ("biasGradient", ctypes.c_double),
        ]

class CGradientVector(ctypes.Structure):
    '''
        Python copy representing the GradientVector struct in the shared library
    '''
    def __init__(self):
        self._fields_ = [
            ("nrOfNeurons", ctypes.c_int),
            ("gradients", ctypes.POINTER(ctypes.POINTER(CNeuronGradient)))
        ]

class CBatch(ctypes.Structure):
    '''
        Python copy representing the Batch struct in the shared library
    '''
    def __init__(self):
        self._fields_ = [
            ("batchSize", ctypes.c_int),
            ("gradientVectors", ctypes.POINTER(ctypes.POINTER(CGradientVector)))
        ]

class NeuralNetwork:
    '''
        A Python object wrapper for a C NeuralNetwork struct from the shared library
    '''
    def __init__(self):
        self.c_nn_ptr = ctypes.pointer(CNeuralNetwork)
        


        
        