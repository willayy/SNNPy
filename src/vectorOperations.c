#include "vectorOperations.h"
#include <stdlib.h>

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
double * vectorMul(double * a, double b, int vectorSize) {
    double * resultVector = (double*)malloc(vectorSize * sizeof(double));

    for (int i = 0; i < vectorSize; i++) {
        resultVector[i] = a[i] * b;
    }

    return resultVector;
}

/**
 * Adds vector b to vector a.
 * @param a The first vector.
 * @param b The second vector.
 * @param vectorSize The size of the vectors. */
void vectorAdd(double * a, double * b, int vectorSize) {
    double * resultVector = (double*)malloc(vectorSize * sizeof(double));

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