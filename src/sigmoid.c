#include "sigmoid.h"
#include <math.h>


static double e = 2.7182818284590452353602874;


/**
 * The sigmoid function (1 / (1 + e^-x)
 * @param x The input value
 * @return The sigmoid of the input value (lower bound 0, upper bound 1)
*/
double sigmoid(double x) {   
    return 1 / (1 + pow(e,-x));
}