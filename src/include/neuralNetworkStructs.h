#include "funcPtrs.h"

#ifndef neuronStructs_h
    #define neuronStructs_h

        /**
         * A Neural network implemented as a struct.
         * These are the following members and their purpose:
         * 
         * @param nrOfParameterNeurons: the number of parameter neurons in the network.
         * @param nrOfHiddenNeurons: the number of hidden neurons in the network.
         * @param nrOfOutputNeurons: the number of output neurons in the network.
         * @param nrOfNeurons: the total number of neurons in the network.
         * @param nrOfHiddenLayers: the number of hidden layers in the network.
         * @param neuronsPerLayer: the number of neurons per hidden layer in the network.
         * 
         * @param activationFunction: the activation function used in the hidden layers of the network.
         * @param lastLayerActivationFunction: the activation function used in the output layer of the network.
         * @param activationFunctionDerivative: the derivative of the activation function used in the hidden layers of the network.
         * @param lastLayerActivationFunctionDerivative: the derivative of the activation function used in the output layer of the network.
         * 
         * @param neuronActivationVector: the activation vector of the neurons in the network.
         * @param hiddenActivationVector: the activation vector of the hidden neurons in the network.
         * @param activationOutputVector: the activation vector of the output neurons in the network.
         * @param activationParameterVector: the activation vector of the parameter neurons in the network.
         * 
         * @param neuronValueVector: the value vector of the neurons in the network.
         * @param hiddenValueVector: the value vector of the hidden neurons in the network.
         * @param outputValueVector: the value vector of the output neurons in the network.
         * @param parameterValueVector: the value vector of the parameter neurons in the network.
         * 
         * The value is the sum of inputs to the neuron, and the activation is the value passed through the activation function.
         * 
         * @param weightMatrix: the weight matrix of the network.
         * @param biasVector: the bias vector of the network. */
        struct NeuralNetwork {

            int nrOfParameterNeurons;
            int nrOfHiddenNeurons;
            int nrOfOutputNeurons;
            int nrOfNeurons;
            int nrOfHiddenLayers;
            int neuronsPerLayer;

            dblAdblR activationFunction;
            dblAdblR lastLayerActivationFunction;
            dblAdblR activationFunctionDerivative;
            dblAdblR lastLayerActivationFunctionDerivative;

            double * neuronActivationVector;
            double * hiddenActivationVector;
            double * activationOutputVector;
            double * activationParameterVector;

            double * neuronValueVector;
            double * hiddenValueVector;
            double * outputValueVector;
            double * parameterValueVector;
    
            double ** weightMatrix;
            double * biasVector;
        };
        typedef struct NeuralNetwork NeuralNetwork;

        /**
         * A struct to hold the gradients of the weights and biases of a neuron.
         * @param weightGradient: the gradient of the weights of the neuron.
         * @param biasGradient: the gradient of the bias of the neuron. */
        struct NeuronGradient {
            double * weightGradient;
            double biasGradient;
        };
        typedef struct NeuronGradient NeuronGradient;


#endif