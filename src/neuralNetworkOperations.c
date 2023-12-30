#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "neuralNetworkOperations.h"
#include "sigmoid.h"
#include <stdlib.h>

/**
 * Calculates the output of the neural network from a double array input.
 * @param nn The neural network to calculate the output of.
 * @param inputData The input data to the neural network. Must be of the same size as the input layer of the neural network.
*/
double * inputDataToNeuralNetwork(struct NeuralNetwork nn, double * inputData) {
    
    // input data into parameter layer
    for (int i = 0; i < nn.nrOfParameters; i++) {
        nn.parameterLayer.parameters[i] = sigmoid(inputData[i]);
    }
    
    // input data into first intermediate layer
    free(nn.intermediateLayers[0].output);
    nn.intermediateLayers[0].output = vectorMatrixMul(nn.parameterLayer.parameters, nn.intermediateLayers[0].edges, nn.nrOfParameters, nn.neuronsPerLayer);

    for (int i = 0; i < nn.neuronsPerLayer; i++ ) {
        nn.intermediateLayers[0].output[i] = sigmoid(nn.intermediateLayers[0].output[i]);
    }

    // input data into intermediate layers and propagate forward
    for (int i = 1; i < nn.nrOfLayers; i++) {

        free(nn.intermediateLayers[i].output); // free output from previous input
        nn.intermediateLayers[i].output = vectorMatrixMul(nn.intermediateLayers[i - 1].output, nn.intermediateLayers[i].edges, nn.neuronsPerLayer, nn.neuronsPerLayer);
        double * temp = nn.intermediateLayers[i].output; // save output without added biases to free it later
        nn.intermediateLayers[i].output = vectorAdd(nn.intermediateLayers[i].output, nn.intermediateLayers[i].biases, nn.neuronsPerLayer);
        free(temp);

        for (int j = 0; j < nn.neuronsPerLayer; j++) {
            nn.intermediateLayers[i].output[j] = sigmoid(nn.intermediateLayers[i].output[j]);
        }
    }

    // input data into output layer
    free(nn.outputLayer.output); // free output from previous input
    nn.outputLayer.output = vectorMatrixMul(nn.intermediateLayers[nn.nrOfLayers - 1].output, nn.outputLayer.edges, nn.neuronsPerLayer, nn.nrOfOutputs);

    for (int i = 0; i < nn.nrOfOutputs; i++) {
        nn.outputLayer.output[i] = sigmoid(nn.outputLayer.output[i]);
    }

    return nn.outputLayer.output;
}
