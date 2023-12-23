#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

    struct NeuralNetwork createNeuralNetwork(int nrOfParameters, int nrOfLayers, int neuronsPerLayer);

    void freeNeuralNetwork(struct NeuralNetwork neuralNetwork);

#endif