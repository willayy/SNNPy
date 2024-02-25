#include <stdlib.h>

#include "neuralNetworkTraining.h"

#include "neuralNetworkOperations.h"
#include "neuralNetworkStructs.h"
#include "vectorOperations.h"
#include "costFunctions.h"
#include "activationFunctions.h"
#include "neuralNetworkUtility.h"
#include "funcPtrs.h"


computeGradientsWeights(struct NeuralNetwork * nn, double * desiredOutput, dblA_dblA costFunctionDerivative, double ** gradients) {
    
    // The derivatives of the cost function with respect to the activations of the output layer.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons; i < nn->nrOfNeurons; i++) {
        double * neuronA = findNeuronActivation(nn, i);
        double * neuronZ = findNeuronValue(nn, i);
        double dCdA = costFunctionDerivative(neuronA[0], desiredOutput[i]);
        double dAdZ = nn->lastLayerActivationFunctionDerivative(neuronZ[0]);
        neuronA[0] = dCdA * dAdZ;
    }

    // the derivatives of the cost function with respect to the activations of the hidden layers.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons - nn->neuronsPerLayer; i > 0; i--) {
        double dZdA;
        double dAdZ;
        double * neuronA = findNeuronActivation(nn, i);
        double * neuronZ = findNeuronValue(nn, i);
        double * connectedneurons = findConnectedNeuronActivations(nn, i);
        double * weights = findOutputWeights(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        double gradientSum = 0;
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            dZdA = connectedneurons[j] * weights[j];
            dAdZ = nn->activationFunctionDerivative(neuronZ[0]);
            gradientSum += dZdA * dAdZ;
        }
        neuronA[0] = gradientSum;
    }

    // the derivatives of the cost function with respect to the weights of the second to last layer.
    for (int i = nn->nrOfNeurons - nn->nrOfOutputNeurons - nn->neuronsPerLayer; i < nn->nrOfNeurons - nn->nrOfOutputNeurons; i++) {
        double * gradientVector = gradients[i];
        double * dCdAxdAdZ = findConnectedNeuronActivations(nn, i);
        double * weights = findOutputWeights(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            double dZdW = weights[j];
            gradientVector[j] = dCdAxdAdZ[j] * dZdW;
        }
    }

    // the derivatives of the cost function with respect to the weights first to second to last layer.
    for (int i = 0; i < nn->nrOfNeurons - nn->nrOfOutputNeurons - nn->neuronsPerLayer; i++) {
        double dZdW;        
        double * gradientVector = gradients[i];
        double * dZdAxdAdZ = findConnectedNeuronActivations(nn, i);
        double * weights = findOutputWeights(nn, i);
        int nrOfConnectedNeurons = numberOfConnectedNeurons(nn, i);
        for (int j = 0; j < nrOfConnectedNeurons; j++) {
            dZdW = weights[j];
            gradientVector[j] = dZdAxdAdZ[j] * dZdW;
        }
    }

}

void nudgeWeight(double * weight, double gradient, double lrw) {

    weight[0] -= lrw * gradient; 
}

void nudgeBias(double * bias, double gradient, double lrb) {

    bias[0] -= lrb * gradient;
}

double ** computeWeightGradient(struct NeuralNetwork * nn, double * desiredOutput, dblA_dblA costFunctionDerivative) {

    double ** gradients = (double **) malloc(sizeof(double *) * nn->nrOfNeurons - nn->nrOfOutputNeurons);
    for (int i = 0; i < nn->nrOfNeurons; i++) {
        gradients[i] = (double *) malloc(sizeof(double) * numberOfConnectedNeurons(nn, i));
    }

    outputLayerDerivatives(nn, desiredOutput, costFunctionDerivative);
    
    hiddenLayerDerivatives(nn);

    return nn->neuronActivationVector;
}


void fit(struct NeuralNetwork * nn, double * gradients, double lrw, double lrb) {
    
    for (int i = 0; i < nn->nrOfNeurons; i++) {
        nn->neuronActivationVector[i] = gradients[i];
    }

    optimize(nn, gradients, lrw, lrb);
}
