#include "neuralNetworkStructs.h"
#include "neuralNetworkUtility.h"
#include "neuralNetworkInit.h"
#include "vectorOperations.h"
#include "randomValueGenerator.h"
#include "testing.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main() {

    int testSumUtility = 0;

    struct NeuralNetwork * nn = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 1, 1, 1, 1, (double[]){-1, 1}, (double[]){-1, 1}, time(NULL));
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,0), 1, "testNr 0, numberOfConnectedNeurons 1, 1, 1, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,1), 1, "testNr 1, numberOfConnectedNeurons 1, 1, 1, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,2), 0, "testNr 2, numberOfConnectedNeurons 1, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,0), "testNr 3, isNeuronLastInLayer 1, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,1), "testNr 4, isNeuronLastInLayer 1, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,2), "testNr 5, isNeuronLastInLayer 1, 1, 1, 1 network");
    freeNeuralNetwork(nn);

    nn = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 1, 2, 3, 1, (double[]){-1, 1}, (double[]){-1, 1}, time(NULL));
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,0), 3, "testNr 6, numberOfConnectedNeurons 1, 2, 3, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,3), 3, "testNr 7, numberOfConnectedNeurons 1, 2, 3, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,6), 1, "testNr 8, numberOfConnectedNeurons 1, 2, 3, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,7), 0, "testNr 9, numberOfConnectedNeurons 1, 2, 3, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,0), "testNr 10, isNeuronLastInLayer 1, 2, 3, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,3), "testNr 11, isNeuronLastInLayer 1, 2, 3, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,6), "testNr 12, isNeuronLastInLayer 1, 2, 3, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,7), "testNr 13, isNeuronLastInLayer 1, 2, 3, 1 network");
    freeNeuralNetwork(nn);

    nn = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 2, 2, 2, 2, (double[]){-1, 1}, (double[]){-1, 1}, time(NULL));
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,0), 2, "testNr 14, numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,2), 2, "testNr 15, numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,4), 2, "testNr 16, numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,5), 2, "testNr 17, numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,6), 0, "testNr 18, numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,7), 0, "testNr 19, numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,1), "testNr 20, isNeuronLastInLayer 2, 2, 2, 2 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,3), "testNr 21, isNeuronLastInLayer 2, 2, 2, 2 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,5), "testNr 22, isNeuronLastInLayer 2, 2, 2, 2 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,7), "testNr 23, isNeuronLastInLayer 2, 2, 2, 2 network");
    freeNeuralNetwork(nn);    

    nn = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 5, 1, 1, 1, (double[]){-1, 1}, (double[]){-1, 1}, time(NULL));
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,0), 1, "testNr 24, numberOfConnectedNeurons 5, 1, 1, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,4), 1, "testNr 25, numberOfConnectedNeurons 5, 1, 1, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,5), 1, "testNr 26, numberOfConnectedNeurons 5, 1, 1, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,6), 0, "testNr 27, numberOfConnectedNeurons 5, 1, 1, 1 network");
    testSumUtility += assertFalse(isNeuronLastInLayer(nn,0), "testNr 28, isNeuronLastInLayer 5, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,4), "testNr 29, isNeuronLastInLayer 5, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,5), "testNr 30, isNeuronLastInLayer 5, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,6), "testNr 31, isNeuronLastInLayer 5, 1, 1, 1 network");
    freeNeuralNetwork(nn);

    nn = (struct NeuralNetwork *) malloc(sizeof(struct NeuralNetwork));
    initNeuralNetwork(nn, 1, 1, 1, 5, (double[]){-1, 1}, (double[]){-1, 1}, time(NULL));
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,0), 1, "testNr 32, numberOfConnectedNeurons 1, 1, 1, 5 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,1), 5, "testNr 33, numberOfConnectedNeurons 1, 1, 1, 5 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,2), 0, "testNr 34, numberOfConnectedNeurons 1, 1, 1, 5 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,0), "testNr 35, isNeuronLastInLayer 1, 1, 1, 5 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,1), "testNr 36, isNeuronLastInLayer 1, 1, 1, 5 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,6), "testNr 37, isNeuronLastInLayer 1, 1, 1, 5 network");
    freeNeuralNetwork(nn);

    printf("\nRunning tests for vectorOperations.c functions\n\n");

    int testSumVectorOperations = 0;

    double * vector1 = (double *) malloc(sizeof(double) * 3);
    double * vector2 = (double *) malloc(sizeof(double) * 3);
    double * vector3 = (double *) malloc(sizeof(double) * 3);
    vector1[0] = 1; vector2[0] = 1; vector3[0] = 1;
    vector1[1] = 2; vector2[1] = 2; vector3[1] = 1;
    vector1[2] = 3; vector2[2] = 3; vector3[2] = 1;


    testSumVectorOperations += dbl_assertEqual(dotProduct(vector1, vector2, 3), 14, "testNr 38, vectorDotProduct (1, 2, 3), (1, 2, 3)");
    testSumVectorOperations += dbl_assertEqual(dotProduct(vector1, vector2, 2), 5, "testNr 39, vectorDotProduct (1, 2), (1, 2)");
    testSumVectorOperations += dbl_assertEqual(dotProduct(vector1, vector2, 1), 1, "testNr 40, vectorDotProduct (1), (1)");
    double * vectorCpy = vectorMulCopy(vector1, 2, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 2, "testNr 41, vectorMulCopy (1, 2, 3) * 2");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 4, "testNr 42, vectorMulCopy (1, 2, 3) * 2");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 6, "testNr 43, vectorMulCopy (1, 2, 3) * 2");
    free(vectorCpy);
    vectorCpy = vectorCopy(vector1, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 1, "testNr 44, vectorCopy (1, 2, 3)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 2, "testNr 45, vectorCopy (1, 2, 3)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 3, "testNr 46, vectorCopy (1, 2, 3)");
    vectorExtend(vectorCpy, vector3, 0, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 1, "testNr 47, vectorExtend (1, 2, 3) with (1, 1, 1) from index 0");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 1, "testNr 48, vectorExtend (1, 2, 3) with (1, 1, 1) from index 0");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 1, "testNr 49, vectorExtend (1, 2, 3) with (1, 1, 1) from index 0");
    vectorAdd(vectorCpy, vector3, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 2, "testNr 50, vectorAdd (1, 1, 1) with (1, 1, 1)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 2, "testNr 51, vectorAdd (1, 1, 1) with (1, 1, 1)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 2, "testNr 52, vectorAdd (1, 1, 1) with (1, 1, 1)");
    vectorReplace(vectorCpy, vector2, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 1, "testNr 53, vectorReplace (1, 1, 1) with (1, 2, 3)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 2, "testNr 54, vectorReplace (1, 1, 1) with (1, 2, 3)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 3, "testNr 55, vectorReplace (1, 1, 1) with (1, 2, 3)");
    vectorMul(vectorCpy, 9, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 9, "testNr 56, vectorMul (1, 2, 3) with 9");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 18, "testNr 57, vectorMul (1, 2, 3) with 9");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 27, "testNr 58, vectorMul (1, 2, 3) with 9");
    free(vectorCpy);
    free(vector1);
    free(vector2);
    free(vector3);

    printf("\nRunning tests for randomValueGenerator.c to check for good distribution\n\n");

    int testSumRandomValueGenerator = 0;

    setSeed(time(NULL));
    double sum = 0;
    for (int i = 0; i < 10000; i++) {
        sum += randomValue(-10, 10);
    }
    double average = sum / 10000;
    testSumRandomValueGenerator += dbl_assertBetween(-0.1, 0.1, average, "testNr 59, randomValue -10, 10 average 0");

    printf("\nRunning tests for neuralNetworkUtility.c functions\n\n");

    return testSumUtility + testSumVectorOperations + testSumRandomValueGenerator;
}