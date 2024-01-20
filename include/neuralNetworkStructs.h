#ifndef neuronStructs_h
    #define neuronStructs_h

    /**
     * A neural network
     * @param nrOfParameters: the number of parameters in the network.
     * @param nrOfLayers: the number of layers in the network.
     * @param neuronsPerLayer: the number of neurons per layer in the network.
     * @param nrOfOutputs: the number of outputs in the network.
     * @param nrOfWeights: the number of weights in the network.
     * @param weightsPerLayer: the number of weights per hidden layer in the network.
     * @param parameterVector: the vector of parameters in the network.
     * @param weightMatrix: the matrix of weights in the network.
     * @param biasVector: the vector of biases in the network.
     * @param outputVector: the vector of outputs in the network.
     * @param neuronVector: the vector of neurons in the network. */
    struct NeuralNetwork {
        int nrOfParameters;
        int nrOfLayers;
        int neuronsPerLayer;
        int nrOfOutputs;
        int nrOfWeights;
        int weightsPerLayer;
        int nrOfHiddenNodes;
        double * parameterVector;
        double * weightMatrix;
        double * biasVector;
        double * outputVector;
        double * neuronVector;
    };

#endif