#include <stdlib.h>
#include "neuralNetworkFuncs.h"
#include "neuralNetworkStructs.h"

/**      
 * Creates a parameter layer.
 * @param nrOfParameters: the number of parameters in the layer. */
struct ParameterLayer createParameterLayer(int nrOfParameters) {
    struct ParameterLayer parameterLayer;
    parameterLayer.size = nrOfParameters;

    parameterLayer.parameters = (double *) malloc(nrOfParameters * sizeof(double));

    for (int i = 0; i < nrOfParameters; i++) {
        parameterLayer.parameters[i] = 69.00;
    }

    return parameterLayer;
}

/**
 * Creates a neuron layer.
 * @param edgesPerNeuron: the number of edges per neuron. This should be the same as the neuron count of the previous layer.
 * @param nrOfNeurons: the number of neurons in the layer. */
struct NeuronLayer createNeuronLayer(int edgesPerNeuron, int nrOfNeurons) {
    struct NeuronLayer neuronLayer;
    neuronLayer.size = nrOfNeurons;

    neuronLayer.edges = (double *) malloc(edgesPerNeuron * nrOfNeurons * sizeof(double));
    neuronLayer.output = (double *) malloc(nrOfNeurons * sizeof(double));

    for (int i = 0; i < edgesPerNeuron * nrOfNeurons; i++) {
        neuronLayer.edges[i] = 69.00;
    }
    for (int i = 0; i < nrOfNeurons; i++) {
        neuronLayer.output[i] = 69.00;
    }

    return neuronLayer;
}

/**
 * Frees a neuron layer from memory.
 * @param neuronLayer: the neuron layer to free from memory. 
*/
void freeNeuronLayer(struct NeuronLayer neuronLayer) {
    free(neuronLayer.edges);
    free(neuronLayer.output);
}

/**
 * Frees a parameter layer from memory.
 * @param parameterLayer: the parameter layer to free from memory. 
*/
void freeParameterLayer(struct ParameterLayer parameterLayer) {
    free(parameterLayer.parameters);
}