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


class NeuralNetwork:
    '''
        A Python object wrapper for a C NeuralNetwork struct from the shared library
    '''
    def __init__(self):
        self.c_nn_ptr = ctypes.pointer(CNeuralNetwork)
        


        
        