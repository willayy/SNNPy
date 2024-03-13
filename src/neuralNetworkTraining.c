#include <stdlib.h>

#include "neuralNetworkTraining.h"

#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunctions.h"
#include "activationFunctions.h"
#include "neuralNetworkUtility.h"
#include "funcPtrs.h"

void initGradient(NeuronGradient * ng, int nrOfConnectedNeurons) {
    ng->nrOfWeights = nrOfConnectedNeurons;
    ng->weightGradient = (double *) malloc(sizeof(double) * nrOfConnectedNeurons);
    ng->biasGradient = (double *) malloc(sizeof(double));
    vectorSet(ng->weightGradient, 0, nrOfConnectedNeurons);
    ng->biasGradient[0] = 0;
}

void initGradientVector(GradientVector * gv, int size) {
    gv->nrOfNeurons = size;
    gv->gradients = (NeuronGradient **) malloc(sizeof(NeuronGradient *) * (gv->nrOfNeurons));
    for (int i = 0; i < gv->nrOfNeurons; i++) {
        gv->gradients[i] = NULL;
    }
}

void initGradientBatch(GradientBatch * gb, int batchSize) {
    gb->batchSize = batchSize;
    gb->gradientVectors = (GradientVector **) malloc(sizeof(GradientVector *) * batchSize);
    for (int i = 0; i < batchSize; i++) {
        gb->gradientVectors[i] = NULL;
    }
}

GradientVector * computeGradients(NeuralNetwork * nn, double * partialGradients) {

    GradientVector * gv = (GradientVector *) malloc(sizeof(GradientVector));

    initGradientVector(gv, nn->nrOfNeurons);

    // the derivatives of the cost function with respect to the weights and biases from first to second to last layer.
    for (int i = 0; i < gv->nrOfNeurons; i++) {
        int * partialGradientIndexes = findConnectedNeuronIndexes(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        NeuronGradient * ng = (NeuronGradient *) malloc(sizeof(NeuronGradient));
        initGradient(ng, nrOfConnectedNeurons);
        gv->gradients[i] = ng;
        double dZdW = nn->neuronActivationVector[i];
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            int pgIndex = partialGradientIndexes[j];
            gv->gradients[i]->weightGradient[j] = (partialGradients[pgIndex] * dZdW);
        }
        gv->gradients[i]->biasGradient[0] = partialGradients[i];
    }

    return gv;
}

double * computePartialGradients(NeuralNetwork * nn, double * desiredOutput, dblAdbLAdblR costFunctionDerivative) {

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

GradientVector * averageGradients(GradientBatch * gb) {

    GradientVector * gv = (GradientVector *) malloc(sizeof(GradientVector));

    initGradientVector(gv, gb->gradientVectors[0]->nrOfNeurons);

    const double avg = 1.0/gb->batchSize;

    for (int i = 0; i < gb->batchSize; i++) {
        for (int j = 0; j < gv->nrOfNeurons; j++) {
            NeuronGradient * ng = (NeuronGradient *) malloc(sizeof(NeuronGradient));
            initGradient(ng, gb->gradientVectors[i]->gradients[j]->nrOfWeights);
            gv->gradients[j] = ng;
            for (int k = 0; k < gv->gradients[j]->nrOfWeights; k++) {
                gv->gradients[j]->weightGradient[k] += (gb->gradientVectors[i]->gradients[j]->weightGradient[k])*avg;
            }
            gv->gradients[j]->biasGradient[0] += (gb->gradientVectors[i]->gradients[j]->biasGradient[0]) * avg;
        }
    }

    return gv;
}

void optimize(NeuralNetwork * nn, GradientVector * avgNg, double lrw, double lrb) {

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        double * weights = findOutputWeights(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double gradient = avgNg->gradients[i]->weightGradient[j];
            weights[j] -= lrw * gradient;
        }
        double gradient = avgNg->gradients[i]->biasGradient[0];
        nn->biasVector[i] -= lrb * gradient;
    }

}