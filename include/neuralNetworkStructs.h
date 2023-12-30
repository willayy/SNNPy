#ifndef neuronStructs_h
    #define neuronStructs_h

    /**
     * A layer of parameters.
     * @param size: the number of parameters in the layer.
     * @param parameters: the double values of the parameters in the layer. */
    struct ParameterLayer {
        int size;
        double * parameters;
    };

    /**
     * A hidden layer of neurons.
     * @param size: the number of neurons in the layer.
     * @param edges: the weights of the edges between the neurons and the previous layer.
     * @param output: the output double values of the neurons in the layer. */
    struct NeuronLayer {
        int size;
        double * edges;
        double * biases;
        double * output;
    };

    /**
     * A neural network
     * @param nrOfParameters: the number of parameters in the network.
     * @param nrOfLayers: the number of layers in the network.
     * @param neuronsPerLayer: the number of neurons per layer in the network.
     * @param nrOfOutputs: the number of outputs in the network.
     * @param parameter: the parameter layer of the network.
     * @param layers: Pointer to the first layer of the network. 
     * @param output: the output layer of the network.
     * */
    struct NeuralNetwork {
        int nrOfParameters;
        int nrOfLayers;
        int neuronsPerLayer;
        int nrOfOutputs;
        struct ParameterLayer parameterLayer;
        struct NeuronLayer * intermediateLayers;
        struct NeuronLayer outputLayer;
    };

#endif