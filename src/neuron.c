#include <stdio.h>
#include <stdlib.h>


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
    double * output;
};

/**
 * A neural network
 * @param amountOfLayers: the number of layers in the network.
 * @param parameter: Pointer to the parameter layer of the network.
 * @param layers: Pointer to the first layer of the network. */
struct NeuralNetwork {
    int nrOfParameters;
    int layersByteSize;
    struct ParameterLayer * parameter;
    struct NeuronLayer * layers;
};

struct NeuralNetwork createNeuralNetwork(int nrOfParameters, int amountOfLayersInBytes) {
    struct NeuralNetwork neuralNetwork;
    neuralNetwork.nrOfParameters = nrOfParameters;
    neuralNetwork.layersByteSize = amountOfLayersInBytes;
    *neuralNetwork.parameter = createParameterLayer(nrOfParameters);
    neuralNetwork.layers = malloc(amountOfLayersInBytes*sizeof(char));
    return neuralNetwork;
}

/**
 * Creates a parameter layer.
 * @param nrOfParameters: the number of parameters in the layer. */
struct ParameterLayer createParameterLayer(int nrOfParameters) {
    struct ParameterLayer parameterLayer;
    parameterLayer.size = nrOfParameters;
    double parameters[nrOfParameters];

    for (int i = 0; i < nrOfParameters; i++) {
        parameters[i] = 0;
    }
    
    parameterLayer.parameters = parameters;
    return parameterLayer;
}

/**
 * Creates a neuron layer.
 * @param edgesPerNeuron: the number of edges per neuron. This should be the same as the neuron count of the previous layer.
 * @param nrOfNeurons: the number of neurons in the layer. */
struct NeuronLayer createNeuronLayer(int edgesPerNeuron, int nrOfNeurons) {
    struct NeuronLayer neuronLayer;
    neuronLayer.size = nrOfNeurons;
    double edges[edgesPerNeuron][nrOfNeurons];
    double output[nrOfNeurons];

    for (int i = 0; i < nrOfNeurons; i++) {
        output[i] = 0;
        for (int j = 0; j < edgesPerNeuron; j++) {
            edges[i][j] = 0;
        }
    }

    neuronLayer.edges = edges;
    neuronLayer.output = output;
    return neuronLayer;
}