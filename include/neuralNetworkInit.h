#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

    struct NeuralNetwork createNeuralNetwork(int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs);

    void freeNeuralNetwork(struct NeuralNetwork neuralNetwork);

#endif