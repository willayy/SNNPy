#ifndef neuron_h_
#define neuron_h_
struct NeuronLayer createNeuronLayer(int edgesPerNeuron, int nrOfNeurons);
struct NeuralNetwork createNeuralNetwork(int nrOfParameters, int amountOfLayersInBytes);
#endif