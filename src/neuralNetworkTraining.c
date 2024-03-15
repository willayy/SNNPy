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
    if (nrOfConnectedNeurons > 0) {
        ng->weightGradient = (double *) malloc(sizeof(double) * nrOfConnectedNeurons);
    } else {
        ng->weightGradient = NULL;
    }
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

/**
 * Computes the gradients for a neural network that has forward propagated a set of inputs.
 * @param nn The neural network to compute the gradients for.
 * @param desiredOutput The desired output of the neural network.
 * @param costFunctionDerivative The derivative of the cost function to use for the output layer.
 * @return A gradient vector with the computed gradients. */
GradientVector * computeGradients(NeuralNetwork * nn, double * desiredOutput, dblAdbLAdblR costFunctionDerivative) {

    double * partialGradients = (double *) malloc(sizeof(double) * nn->nrOfNeurons);
    vectorSet(partialGradients, 0, nn->nrOfNeurons);
    GradientVector * gv = (GradientVector *) malloc(sizeof(GradientVector));
    initGradientVector(gv, nn->nrOfNeurons);
    double neuronA;
    double neuronZ;
    double dCdA;
    double dAdZ;
    double dZdA;
    double dZdW;
    double * outputWeights;
    int * connectedNeuronIndexes;
    int nrOfConnectedNeurons;

    // Calculate dCdA and dAdZ for the output layer.
    for (int i = nn->nrOfNeurons - 1; i > nn->nrOfNeurons - nn->nrOfOutputNeurons - 1; i--) {
        neuronA = nn->neuronActivationVector[i];
        neuronZ = nn->neuronValueVector[i];
        dCdA = costFunctionDerivative(neuronA, desiredOutput[i]);
        dAdZ = nn->lastLayerActivationFunctionDerivative(neuronZ);
        partialGradients[i] = dCdA * dAdZ;
        NeuronGradient * ng = (NeuronGradient *) malloc(sizeof(NeuronGradient));
        initGradient(ng, 0);
        ng->biasGradient[0] = partialGradients[i]; // Bias gradient is the same as the partial gradient. dZdB = 1
        gv->gradients[i] = ng;
    }

    // Calculates dAdZ, dZdA (connected neurons Z value) for hidden layer and parameter neurons. 
    // Multiplies this with connected partial gradients and sums them up.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons - 1; i >= 0; i--) {
        neuronZ = nn->neuronValueVector[i];
        outputWeights = findOutputWeights(nn, i);
        connectedNeuronIndexes = findConnectedNeuronIndexes(nn, i);
        nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        dAdZ = nn->activationFunctionDerivative(neuronZ);

        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            dZdA = outputWeights[j];
            partialGradients[i] += partialGradients[connectedNeuronIndexes[j]] * dZdA * dAdZ;
        }

        NeuronGradient * ng = (NeuronGradient *) malloc(sizeof(NeuronGradient));
        initGradient(ng, nrOfConnectedNeurons);
        dZdW = nn->neuronActivationVector[i];

        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            ng->weightGradient[j] = (partialGradients[connectedNeuronIndexes[j]] * dZdW);
        }

        gv->gradients[i] = ng;
        free(connectedNeuronIndexes);
    }

    free(partialGradients);

    return gv;
}

/**
 * Averages the gradient vectors in a gradient batch.
 * @param gb The gradient batch to average the gradients of.
 * @return A gradient vector with the averaged gradients. */
GradientVector * averageGradients(GradientBatch * gb) {

    GradientVector * gv = (GradientVector *) malloc(sizeof(GradientVector));

    initGradientVector(gv, gb->gradientVectors[0]->nrOfNeurons);

    const double avg = 1.0/((double) gb->batchSize);

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

/**
 * Optimizes the neural network by updating the weights and biases with the gradients.
 * @param nn The neural network to optimize.
 * @param avgNg The averaged gradients to use for optimization.
 * @param lrw The learning rate for the weights.
 * @param lrb The learning rate for the biases. */
void optimize(NeuralNetwork * nn, GradientVector * avgNg, double lrw, double lrb) {
    
    double * connectedWeights;
    int nrOfConnectedNeurons;
    double gradient;

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        connectedWeights = findOutputWeights(nn, i);
        nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            gradient = avgNg->gradients[i]->weightGradient[j];
            connectedWeights[j] -= lrw * gradient;
        }
        gradient = avgNg->gradients[i]->biasGradient[0];
        nn->biasVector[i] -= lrb * gradient;
    }

}