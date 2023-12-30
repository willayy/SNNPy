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
 * calculates the product of a vector multiplied by a matrix.
 * Size of vector must be equal to the number of columns in the matrix.
 * @param vector The vector.
 * @param matrix The matrix.
 * @param vectorSize The size of the vector.
 * @param matrixCols The number of columns in the matrix. */
double * vectorMatrixMul(double * vector, double * matrix, int vectorSize, int matrixCols) {
    double * resultVector = (double*)malloc(vectorSize * sizeof(double));
     
    for (int i = 0; i < vectorSize; i++) {
        resultVector[i] = dotProduct(vector, matrix + i*matrixCols, vectorSize);
    }

    return resultVector;
}

/**
 * Adds two vectors of the same size.
 * @param a The first vector.
 * @param b The second vector.
 * @param vectorSize The size of the vectors. */
double * vectorAdd(double * a, double * b, int vectorSize) {
    double * resultVector = (double*)malloc(vectorSize * sizeof(double));

    for (int i = 0; i < vectorSize; i++) {
        resultVector[i] = a[i] + b[i];
    }

    return resultVector;
}