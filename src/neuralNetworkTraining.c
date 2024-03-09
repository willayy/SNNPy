#include <stdlib.h>

#include "neuralNetworkTraining.h"

#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunctions.h"
#include "activationFunctions.h"
#include "neuralNetworkUtility.h"
#include "funcPtrs.h"

NeuronGradient ** computeGradients(const NeuralNetwork * nn, double * partialGradients) {

    NeuronGradient ** gradients = (double **) malloc(sizeof(NeuronGradient *) * nn->nrOfNeurons);

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        if (i < nn->nrOfNeurons - nn->neuronsPerLayer - nn->nrOfOutputNeurons ) {
            gradients[i]->weightGradient = (double *) malloc(sizeof(double) * nn->neuronsPerLayer);
        } else {
            gradients[i]->weightGradient = (double *) malloc(sizeof(double) * nn->nrOfOutputNeurons);
        }
    }

    // the derivatives of the cost function with respect to the weights first to second to last layer.
    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double * neuronActivations = findConnectedNeuronActivations(nn, i);
        int * partialGradientIndexes = findConnectedNeuronIndexes(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double dZdW = neuronActivations[j];
            gradients[i]->weightGradient[j] = (partialGradients[partialGradientIndexes[j]] * dZdW);
        }
    }

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        gradients[i]->biasGradient = (double *) malloc(sizeof(double));
        gradients[i]->biasGradient[0] = partialGradients[i];
    }

    return gradients;

}

double * computePartialGradients(const NeuralNetwork * nn, double * desiredOutput, dblAdbLAdblR costFunctionDerivative) {

    double * partialGradients = (double *) malloc(sizeof(double) * nn->nrOfNeurons);

    // The derivatives of the cost function with respect to the activations of the output layer.
    for (int i = nn->nrOfNeurons - 1; i > nn->nrOfNeurons - nn->nrOfOutputNeurons - 1; i--) {
        double neuronA = nn->neuronActivationVector[i];
        double neuronZ = nn->neuronValueVector[i];
        double dCdA = costFunctionDerivative(neuronA, desiredOutput[i]);
        double dAdZ = nn->lastLayerActivationFunctionDerivative(neuronZ);
        partialGradients[i] = dCdA * dAdZ;
    }

    // the derivatives of the cost function with respect to the activations of the hidden layers.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons - 1; i >= 0; i--) {
        double neuronZ = nn->neuronValueVector[i];
        double * weights = findOutputWeights(nn, i);
        int * connectedNeuronIndexes = findConnectedNeuronIndexes(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double gradientSum = 0;
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double dZdA = weights[j];
            double dAdZ = nn->activationFunctionDerivative(neuronZ);
            gradientSum += partialGradients[connectedNeuronIndexes[i]] * dZdA * dAdZ;
        }
        partialGradients[i] = gradientSum;
    }

    return partialGradients;
}

NeuronGradient ** averageGradients(const NeuralNetwork * nn, NeuronGradient *** sumNg, int batchSize, int nrOfNeurons) {

    NeuronGradient ** avgNg = (NeuronGradient **) malloc(sizeof(NeuronGradient *) * nrOfNeurons);

    for (int i = 0; i < nrOfNeurons; i++) {
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        avgNg[i]->weightGradient = (double *) malloc(sizeof(double) * nrOfConnectedNeurons);
        avgNg[i]->biasGradient = (double *) malloc(sizeof(double));
    }

    //TODO: implement averaging of gradients.

    return avgNg;
}

void optimize(NeuralNetwork * nn, NeuronGradient ** avgNg, double lrw, double lrb) {

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double * weights = findOutputWeights(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double * weight = &weights[j];
            double gradient = avgNg[i]->weightGradient[j];
            weight[0] -= lrw * gradient;
        }
    }

    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double * bias = &nn->biasVector[i];
        double gradient = avgNg[i]->biasGradient[0];
        bias[0] -= lrb * gradient;
    }
}