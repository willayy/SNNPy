#include "funcPtrs.h"

#ifndef neuronStructs_h
    #define neuronStructs_h

        /**
         * A Neuron implemented as a struct.
         * @param A: the activation of the neuron.
         * @param Z: the value of the neuron.
         * @param activationFunctions: the activation function and its derivative for the neuron.
         * @param connections: the neurons connected to the neuron in the forward direction.
         * @param weights: the weights of the connections.
         * @param bias: the bias of the neuron. */
        struct Neuron {
            double A;
            double Z;
            double bias;
            int connections;
            double * weights;
            dblA_dblR * activationFunctions; // Store activation in [0] and derivative in [1].
            struct Neuron ** connectedNeurons;
        };
        typedef struct Neuron Neuron;

        /**
         * A Neural network implemented as a struct.
         * @param nrOfParameterNeurons: the number of parameter neurons.
         * @param nrOfHiddenNeurons: the number of hidden neurons.
         * @param nrOfOutputNeurons: the number of output neurons.
         * @param nrOfNeurons: the total number of neurons in the network.
         * @param nrOfHiddenLayers: the number of hidden layers in the network.
         * @param neuronsPerLayer: the number of neurons per layer in the network.
         * @param neurons: the neurons in the network. */
        struct NeuralNetwork {

            int nrOfInputNeurons;
            int nrOfHiddenNeurons;
            int nrOfOutputNeurons;
            int nrOfNeurons;
            int nrOfHiddenLayers;
            int neuronsPerLayer;

            dblA_dblR * inputLayerActivationFunctions;
            dblA_dblR * hiddenLayerActivationFunctions;
            dblA_dblR * outputLayerActivationFunctions;

            Neuron ** neurons;
            Neuron ** outputLayer;
        };
        typedef struct NeuralNetwork NeuralNetwork;

        /**
         * A struct to hold the gradients of the weights and biases of a neuron.
         * @param nrOfWeights: the number of weights of the neuron feeding forward.
         * @param weightGradient: the gradient of the weights of the neuron.
         * @param biasGradient: the gradient of the bias of the neuron. */
        struct NeuronGradient {
            int nrOfWeights;
            double * weightGradient;
            double biasGradient;
        };
        typedef struct NeuronGradient NeuronGradient;

        /**
         * A struct to hold the gradients of the weights and biases of a NeuralNetwork.
         * @param nrOfNeurons: the number of neurons in the network that has gradients.
         * @param gradients: the gradients of the weights and biases of the neurons. */
        struct GradientVector {
            int nrOfNeurons;
            NeuronGradient ** gradients;
        };
        typedef struct GradientVector GradientVector;

        /**
         * A struct to hold the gradient vectors of a batch.
         * @param batchSize: the size of the batch.
         * @param gradientVectors: the gradients of the weights and biases of the neurons. */
        struct GradientBatch {
            int batchSize;
            GradientVector ** gradientVectors;
        };
        typedef struct GradientBatch GradientBatch;

        // Needs to be here to avoid recursion in the includes.
        /** pointer for a "double func(NeuralNetwork *)" function */
        typedef double (*nnA_dblR)(NeuralNetwork *);

#endif