#include "../include/neuralNetworkFuncs.h"
#include "../include/neuralNetworkStructs.h"

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