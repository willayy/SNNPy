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
 * Rectified Linear Unit (ReLU) function (max(0, x))
 * @param x The input value
 * @return The ReLU of the input value (lower bound 0, upper bound x) */
double rectifiedLinearUnit(double x) {
    return x > 0 ? x : 0;
}

/**
 * The derivative of the ReLU function
 * @param x The input value
 * @return The derivative of the ReLU function */
double rectifiedLinearUnitDerivative(double x) {
    return x > 0 ? 1 : 0;
}

/**
 * The hyperbolic tangent function (e^x - e^-x) / (e^x + e^-x)
 * @param x The input value
 * @return The hyperbolic tangent of the input value (lower bound -1, upper bound 1) */
double hyperbolicTangent(double x) {
    return (pow(e,x) - pow(e,-x)) / (pow(e,x) + pow(e,-x));
}

/**
 * The derivative of the hyperbolic tangent function (4e^2x / (e^2x + 1)^2)
 * @param x The input value
 * @return The derivative of the hyperbolic tangent function */
double hyperbolicTangentDerivative(double x) {
    return (4*pow(e, 2*x) / pow(pow(e, x*2) + 1, 2));
}
