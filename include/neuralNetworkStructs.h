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
     * @param nrOfWeights: the total number of weights in the network.
     * @param nrOfHiddenLayers: the number of hidden layers in the network.
     * @param neuronsPerLayer: the number of neurons per hidden layer in the network.
     * 
     * @param neuronVector: a vector of all the neurons in the network.
     * @param parameterVector: a pointer to all the parameter neurons in the network (neuronVector + 0).
     * @param hiddenVector: a pointer to all the hidden neurons in the network (neuronVector + nrOfParameterNeurons).
     * @param outputVector: a pointer to all the output neurons in the network (neuronVector + nrOfParameterNeurons + nrOfHiddenNeurons).
     * @param weightMatrix: a matrix of all the weights in the network [i] is the neuron and [j] are the forward connecting weights.
     * @param biasVector: a vector of all the biases in the network.    */  
    struct NeuralNetwork {

        int nrOfParameterNeurons;
        int nrOfHiddenNeurons;
        int nrOfOutputNeurons;
        int nrOfNeurons;
        int nrOfWeights;
        int nrOfHiddenLayers;
        int neuronsPerLayer;

        double * neuronVector;
        double * parameterVector;
        double * hiddenVector;
        double * outputVector;
        double ** weightMatrix;
        double * biasVector;
    };

#endif