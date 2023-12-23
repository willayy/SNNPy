#ifndef neuronFuncs_h
    #define neuronFuncs_h

    struct NeuronLayer createNeuronLayer(int edgesPerNeuron, int nrOfNeurons);

    struct ParameterLayer createParameterLayer(int nrOfParameters);

    struct NeuralNetwork createNeuralNetwork(int nrOfParameters, int nrOfLayers, int neuronsPerLayer);

    void freeNeuronLayer(struct NeuronLayer neuronLayer);

    void freeParameterLayer(struct ParameterLayer parameterLayer);

    void freeNeuralNetwork(struct NeuralNetwork neuralNetwork);

#endif