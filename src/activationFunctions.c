#include "activationFunctions.h"
#include <math.h>


static double e = 2.7182818284590452353602874;


/**
 * The sigmoid function (1 / (1 + e^-x)
 * @param x The input value
 * @return The sigmoid of the input value (lower bound 0, upper bound 1) */
double sigmoid(double x) {   
    return 1 / (1 + pow(e,-x));
}

/**
 * The derivative of the sigmoid function (sigmoid(x) * (1 - sigmoid(x))
 * @param x The input value
 * @return The derivative of the sigmoid function */
double sigmoidDerivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

/**
 * The hyperbolic tangent function tanh(x)
 * @param x The input value
 * @return The hyperbolic tangent of the input value (lower bound -1, upper bound 1) */
double hyperbolicTangent(double x) {
    return (pow(e,x) - pow(e,-x)) / (pow(e,x) + pow(e,-x));
}

/**
 * The derivative of the hyperbolic tangent function d/dx tanh(x)
 * @param x The input value
 * @return The derivative of the hyperbolic tangent function */
double hyperbolicTangentDerivative(double x) {
    return (4*pow(e, 2*x) / pow(pow(e, x*2) + 1, 2));
}