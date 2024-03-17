#include <stdlib.h>

#include "neuralNetworkTraining.h"

#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
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
    for (int i = 0; i < nrOfConnectedNeurons; i++) {ng->weightGradient[i] = 0;}
    ng->biasGradient = 0;
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
    for (int i = 0; i < nn->nrOfNeurons; i++) { partialGradients[i] = 0;}
    GradientVector * gv = (GradientVector *) malloc(sizeof(GradientVector));
    initGradientVector(gv, nn->nrOfNeurons);

    Neuron * n;
    double dCdA;
    double dAdZ;
    double dZdA;
    double dZdW;
    int * connectedNeuronIndexes;

    // Calculate dCdA and dAdZ for the output layer.
    for (int i = nn->nrOfNeurons - 1; i > nn->nrOfNeurons - nn->nrOfOutputNeurons - 1; i--) {
        n = nn->neurons[i];
        dCdA = costFunctionDerivative(n->A, desiredOutput[i]);
        dAdZ = n->activationFunctions[1](n->Z);
        partialGradients[i] = dCdA * dAdZ;
        NeuronGradient * ng = (NeuronGradient *) malloc(sizeof(NeuronGradient));
        initGradient(ng, 0);
        ng->biasGradient = partialGradients[i]; // Bias gradient is the same as the partial gradient. dZdB = 1
        gv->gradients[i] = ng;
    }

    // Calculates dAdZ, dZdA (connected neurons Z value) for hidden layer and parameter neurons. 
    // Multiplies this with connected partial gradients and sums them up.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons - 1; i >= 0; i--) {
        n = nn->neurons[i];
        dAdZ = n->activationFunctions[1](n->Z);
        connectedNeuronIndexes = findConnectedNeuronIndexes(nn, i);

        for (int j = 0; j < n->conections; j++) {
            dZdA = n->weights[j];
            partialGradients[i] += partialGradients[connectedNeuronIndexes[j]] * dZdA * dAdZ;
        }

        NeuronGradient * ng = (NeuronGradient *) malloc(sizeof(NeuronGradient));
        initGradient(ng, n->conections);
        dZdW = n->A;

        
        for (int j = 0; j < n->conections; j++) {
            ng->weightGradient[j] = (partialGradients[connectedNeuronIndexes[j]] * dZdW);
        }

        // Biases in input neurons should remain zero to keep the input data as is
        if (i >= nn->nrOfInputNeurons) {
            ng->biasGradient = partialGradients[i];
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
            gv->gradients[j]->biasGradient += (gb->gradientVectors[i]->gradients[j]->biasGradient) * avg;
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
    
    double gradient;
    Neuron * n;

    for (int i = 0; i < nn->nrOfNeurons; i++) {
        n = nn->neurons[i];
        for (int j = 0; j < n->conections; j++) {
            gradient = avgNg->gradients[i]->weightGradient[j];
            n->weights[j] -= lrw * gradient;
        }
        gradient = avgNg->gradients[i]->biasGradient;
        n->bias -= lrb * gradient;
    }

}