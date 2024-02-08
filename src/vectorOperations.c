#include "vectorOperations.h"
#include <stdlib.h>

/**
 * Copies a vector.
 * @param a The vector.
 * @param vectorSize The size of the vector.
 * @return A copy of the vector. */
double * vectorCopy(double * a, int vectorSize) {
    double * result = malloc(vectorSize * sizeof(double));

    for (int i = 0; i < vectorSize; i++) {
        result[i] = a[i];
    }

    return result;
}

/**
 * Extend a vector A with the elements of vector B from (inclusive) a given index.
 * @param a The first vector.
 * @param b The second vector.
 * @param from The index to start extending from.
 * @param vectorSize The size of the vectors.*/
void vectorExtend(double * a, double * b, int from, int vectorSize) {
    for (int i = 0; i < vectorSize; i++) {
        a[from + i] = b[i];
    }
}

/**
 * Calculates the dot product of two vectors of the same size.
 * @param a The first vector.
 * @param b The second vector.
 * @param vectorSize The size of the vectors. */
double dotProduct(double *a, double *b, int vectorSize) {
    double sum = 0;

    for (int i = 0; i < vectorSize; i++) {
        sum += a[i] * b[i];
    }

    return sum; 
}

/**
 * Multiplies a vector with a scalar.
 * @param a The vector.
 * @param b The scalar.
 * @param vectorSize The size of the vector. */
void vectorMul(double * a, double b, int vectorSize) {

    for (int i = 0; i < vectorSize; i++) {
        a[i] = a[i] * b;
    }

}

/**
 * Multiplies a vector with a scalar and returns a copy of the result.
 * @param a The vector.
 * @param b The scalar.
 * @param vectorSize The size of the vector.
 * @return A copy of the result. */
double * vectorMulCopy(double * a, double b, int vectorSize) {
    double * result = malloc(vectorSize * sizeof(double));

    for (int i = 0; i < vectorSize; i++) {
        result[i] = a[i] * b;
    }

    return result;
}

/**
 * Adds vector b to vector a.
 * @param a The first vector.
 * @param b The second vector.
 * @param vectorSize The size of the vectors. */
void vectorAdd(double * a, double * b, int vectorSize) {
    for (int i = 0; i < vectorSize; i++) {
        a[i] = a[i] + b[i];
    }
}

/**
 * Replaces vector elements in a with vector b elements.
 * @param a The first vector.
 * @param b The second vector.
 * @param vectorSize The size of the vectors. */
void vectorReplace(double * a, double * b, int vectorSize) {
    for (int i = 0; i < vectorSize; i++) {
        a[i] = b[i];
    }
}

/**
 * Applies a function to each element of a vector. Function must take a double as a parameter and return an double.
 * @param a The vector.
 * @param vectorSize The size of the vector.
 * @param operation The function to be applied to each element of the vector. */
void vectorOperation(double * a, singeleDoubleParamOperation operation, int vectorSize) {
    for (int i = 0; i < vectorSize; i++) {
        a[i] = operation(a[i]);
    }
}