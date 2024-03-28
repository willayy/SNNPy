#include "costFunctions.h"
#include "neuralNetworkStructs.h"
#include <math.h>
#include <stdlib.h>

/**
 * Calculates the mean square cost of the neural network on a given input and desired output. 
 * @param output The output of the neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param outputSize The size of the output array.
 * @return The cost of the neural network on the given input and desired output. */
double sqrCostFunction(double * output, double * desiredOutput, int outputSize) {

    // cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
    double cost = 0;
    double error;
    for (int i = 0; i < outputSize; i++) {
        error = output[i] - desiredOutput[i];
        cost += error * error;
    }
    return 0.5 * cost;
}

/**
 * Calculates the derivative of the mean square cost function for a single output.
 * @param output The output of the neural network neuron.
 * @param desiredOutput The desired output of the neural network.
 * @return The derivative of the cost function. */
double sqrCostFunctionDerivative(double output, double desiredOutput) {
    return (output-desiredOutput);
}

/**
 * Calculates the cross entropy cost of the neural network on a given input and desired output.
 * Works best for binary classification.
 * @param output The output of the neural network.
 * @param desiredOutput The desired output of the neural network.
 * @param outputSize The size of the output array.
 * @return The cost of the neural network on the given input and desired output. */
double crossEntropyCostFunction(double * output, double * desiredOutput, int outputSize) {

    // Note: expected outputs are expected to all be either 0 or 1
    // cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
    double cost = 0;
    double x;
    double y;
    double v;
    for (int i = 0; i < outputSize; i++) {
        x = output[i];
        y = desiredOutput[i];
        v = (y == 1) ? -log(x) : -log(1 - x);
        cost += isnan(v) ? 0 : v;
    }
    return cost;
}

/**
 * Calculates the derivative of the cross entropy cost function for a single output.
 * @param output The output of the neural network neuron.
 * @param desiredOutput The desired output of the neural network.
 * @return The derivative of the cost function. */
double crossEntropyCostFunctionDerivative(double output, double desiredOutput ) {
    double x = output;
    double y = desiredOutput;
    if (x == 0 || x == 1) { return 0; }
    return (-x + y) / (x * (x - 1));
}

double noRegularization(Neuron ** nv, int nrOfNeurons) {
    return 0;
}

double noRegularizationDerivative(double weight) {
    return 0;
}

double l1Regularization(Neuron ** nv, int nrOfNeurons) {
    double sum = 0;
    double weight;
    for (int i = 0; i < nrOfNeurons; i++) {
        for (int j = 0; j < (nv[i])->connections; j++) {
            weight = nv[i]->weights[j];
            sum += fabs(weight);
        }
    }
    return sum;
}

double l2Regularization(Neuron ** nv, int nrOfNeurons) {
    double sum = 0;
    double weight;
    for (int i = 0; i < nrOfNeurons; i++) {
        for (int j = 0; j < (nv[i])->connections; j++) {
            weight = nv[i]->weights[j];
            sum += weight * weight;
        }
    }
    return sum;
}

double l1RegularizationDerivative(double weight) {
    if (weight == 0) { return 0; }
    return (weight > 0) ? 1 : -1;
}

double l2RegularizationDerivative(double weight) {
    return 2*weight;
}