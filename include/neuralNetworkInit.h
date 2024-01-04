#ifndef neuralNetworkInit_h
    #define neuralNetworkInit_h

    struct NeuralNetwork createNeuralNetwork(int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs);

    void resetNeuralNetwork(struct NeuralNetwork nn);

    void freeNeuralNetwork(struct NeuralNetwork nn);

#endif