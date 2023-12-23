#include <stdlib.h>
#include "neuralNetworkInit.h"
#include "neuralNetworkStructs.h"

/**      
 * Creates a parameter layer.
 * @param nrOfParameters: the number of parameters in the layer. */
struct ParameterLayer createParameterLayer(int nrOfParameters) {
    struct ParameterLayer parameterLayer;
    parameterLayer.size = nrOfParameters;

    parameterLayer.parameters = (double *) malloc(nrOfParameters * sizeof(double));

    for (int i = 0; i < nrOfParameters; i++) {
        parameterLayer.parameters[i] = 1.00;
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
        neuronLayer.edges[i] = 1.00;
    }
    for (int i = 0; i < nrOfNeurons; i++) {
        neuronLayer.output[i] = 1.00;
    }

    return neuronLayer;
}

/**
 * Creates a neural network.
 * @param nrOfParameters: the number of parameters in the network.
 * @param nrOfLayers: the number of layers in the network.
 * @param neuronsPerLayer: the number of neurons per layer in the network.
 * @param nrOfOutputs: the number of outputs in the network. */
struct NeuralNetwork createNeuralNetwork(int nrOfParameters, int nrOfLayers, int neuronsPerLayer, int nrOfOutputs) {
    struct NeuralNetwork neuralNetwork;

    neuralNetwork.nrOfParameters = nrOfParameters;
    neuralNetwork.nrOfLayers = nrOfLayers;
    neuralNetwork.neuronsPerLayer = neuronsPerLayer;
    neuralNetwork.nrOfOutputs = nrOfOutputs;

    neuralNetwork.parameterLayer = createParameterLayer(nrOfParameters);

    neuralNetwork.intermediateLayers = (struct NeuronLayer *) malloc((nrOfLayers * neuronsPerLayer * neuronsPerLayer)*sizeof(double));

    for (int i = 0; i < nrOfLayers; i++) {
        neuralNetwork.intermediateLayers[i] = createNeuronLayer(neuronsPerLayer, neuronsPerLayer);
    }

    neuralNetwork.outputLayer = createNeuronLayer(neuronsPerLayer, nrOfOutputs);

    return neuralNetwork;
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

/**
 * Frees a neural network from memory.
 * @param neuralNetwork: the neural network to free from memory. 
*/
void freeNeuralNetwork(struct NeuralNetwork neuralNetwork) {
    freeParameterLayer(neuralNetwork.parameterLayer);
    int i = 0;
    for (int _ = 0; _ < neuralNetwork.nrOfLayers; _++) {
        for (int _ = 0; _ < neuralNetwork.neuronsPerLayer; _++) {
            freeNeuronLayer(neuralNetwork.intermediateLayers[i]);
            i++;
        }
    }
}