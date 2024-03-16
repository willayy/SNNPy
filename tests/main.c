#include "neuralNetworkStructs.h"
#include "neuralNetworkUtility.h"
#include "neuralNetworkInit.h"
#include "vectorOperations.h"
#include "randomValueGenerator.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "testing.h"
#include "activationFunctions.h"
#include "costFunctions.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

int main() {

    int testSumUtility = 0;

    printf("\nRunning tests for neuralNetworkUtility.c functions\n\n");

    NeuralNetwork * nn = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    initNeuralNetwork(nn, 1, 1, 1, 1);
    initNeuralNetworkFunctions(nn, &sigmoid, &sigmoidDerivative, &sigmoid, &sigmoidDerivative);
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,0), 1, "numberOfConnectedNeurons 1, 1, 1, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,1), 1, "numberOfConnectedNeurons 1, 1, 1, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,2), 0, "numberOfConnectedNeurons 1, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,0), "isNeuronLastInLayer 1, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,1), "isNeuronLastInLayer 1, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,2), "isNeuronLastInLayer 1, 1, 1, 1 network");
    freeNeuralNetwork(nn);

    nn = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    initNeuralNetwork(nn, 1, 2, 3, 1);
    initNeuralNetworkFunctions(nn, &sigmoid, &sigmoidDerivative, &sigmoid, &sigmoidDerivative);
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,0), 3, "numberOfConnectedNeurons 1, 2, 3, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,3), 3, "numberOfConnectedNeurons 1, 2, 3, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,6), 1, "numberOfConnectedNeurons 1, 2, 3, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,7), 0, "numberOfConnectedNeurons 1, 2, 3, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,0), "isNeuronLastInLayer 1, 2, 3, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,3), "isNeuronLastInLayer 1, 2, 3, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,6), "isNeuronLastInLayer 1, 2, 3, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,7), "isNeuronLastInLayer 1, 2, 3, 1 network");
    freeNeuralNetwork(nn);

    nn = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    initNeuralNetwork(nn, 2, 2, 2, 2);
    initNeuralNetworkFunctions(nn, &sigmoid, &sigmoidDerivative, &sigmoid, &sigmoidDerivative);
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,0), 2, "numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,2), 2, "numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,4), 2, "numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,5), 2, "numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,6), 0, "numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,7), 0, "numberOfConnectedNeurons 2, 2, 2, 2 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,1), "isNeuronLastInLayer 2, 2, 2, 2 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,3), "isNeuronLastInLayer 2, 2, 2, 2 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,5), "isNeuronLastInLayer 2, 2, 2, 2 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,7), "isNeuronLastInLayer 2, 2, 2, 2 network");
    freeNeuralNetwork(nn);    

    nn = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    initNeuralNetwork(nn, 5, 1, 1, 1);
    initNeuralNetworkFunctions(nn, &sigmoid, &sigmoidDerivative, &sigmoid, &sigmoidDerivative);
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,0), 1, "numberOfConnectedNeurons 5, 1, 1, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,4), 1, "numberOfConnectedNeurons 5, 1, 1, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,5), 1, "numberOfConnectedNeurons 5, 1, 1, 1 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,6), 0, "numberOfConnectedNeurons 5, 1, 1, 1 network");
    testSumUtility += assertFalse(isNeuronLastInLayer(nn,0), "isNeuronLastInLayer 5, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,4), "isNeuronLastInLayer 5, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,5), "isNeuronLastInLayer 5, 1, 1, 1 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,6), "isNeuronLastInLayer 5, 1, 1, 1 network");
    freeNeuralNetwork(nn);

    nn = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    initNeuralNetwork(nn, 1, 1, 1, 5);
    initNeuralNetworkFunctions(nn, &sigmoid, &sigmoidDerivative, &sigmoid, &sigmoidDerivative);
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,0), 1, "numberOfConnectedNeurons 1, 1, 1, 5 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,1), 5, "numberOfConnectedNeurons 1, 1, 1, 5 network");
    testSumUtility += int_assertEqual(numberOfConnectedNeurons(nn,2), 0, "numberOfConnectedNeurons 1, 1, 1, 5 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,0), "isNeuronLastInLayer 1, 1, 1, 5 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,1), "isNeuronLastInLayer 1, 1, 1, 5 network");
    testSumUtility += assertTrue(isNeuronLastInLayer(nn,6), "isNeuronLastInLayer 1, 1, 1, 5 network");
    freeNeuralNetwork(nn);

    printf("\nRunning tests for vectorOperations.c functions\n\n");

    int testSumVectorOperations = 0;

    double * vector1 = (double *) malloc(sizeof(double) * 3);
    double * vector2 = (double *) malloc(sizeof(double) * 3);
    double * vector3 = (double *) malloc(sizeof(double) * 3);
    vector1[0] = 1; vector2[0] = 1; vector3[0] = 1;
    vector1[1] = 2; vector2[1] = 2; vector3[1] = 1;
    vector1[2] = 3; vector2[2] = 3; vector3[2] = 1;

    testSumVectorOperations += dbl_assertEqual(dotProduct(vector1, vector2, 3), 14, "vectorDotProduct (1, 2, 3), (1, 2, 3)");
    testSumVectorOperations += dbl_assertEqual(dotProduct(vector1, vector2, 2), 5, "vectorDotProduct (1, 2), (1, 2)");
    testSumVectorOperations += dbl_assertEqual(dotProduct(vector1, vector2, 1), 1, "vectorDotProduct (1), (1)");
    double * vectorCpy = vectorMulCopy(vector1, 2, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 2, "vectorMulCopy (1, 2, 3) * 2");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 4, "vectorMulCopy (1, 2, 3) * 2");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 6, "vectorMulCopy (1, 2, 3) * 2");
    free(vectorCpy);
    vectorCpy = vectorCopy(vector1, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 1, "vectorCopy (1, 2, 3)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 2, "vectorCopy (1, 2, 3)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 3, "vectorCopy (1, 2, 3)");
    vectorExtend(vectorCpy, vector3, 0, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 1, "vectorExtend (1, 2, 3) with (1, 1, 1) from index 0");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 1, "vectorExtend (1, 2, 3) with (1, 1, 1) from index 0");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 1, "vectorExtend (1, 2, 3) with (1, 1, 1) from index 0");
    vectorAdd(vectorCpy, vector3, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 2, "vectorAdd (1, 1, 1) with (1, 1, 1)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 2, "vectorAdd (1, 1, 1) with (1, 1, 1)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 2, "vectorAdd (1, 1, 1) with (1, 1, 1)");
    vectorReplace(vectorCpy, vector2, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 1, "vectorReplace (1, 1, 1) with (1, 2, 3)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 2, "vectorReplace (1, 1, 1) with (1, 2, 3)");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 3, "vectorReplace (1, 1, 1) with (1, 2, 3)");
    vectorMul(vectorCpy, 9, 3);
    testSumVectorOperations += dbl_assertEqual(vectorCpy[0], 9, "vectorMul (1, 2, 3) with 9");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[1], 18, "vectorMul (1, 2, 3) with 9");
    testSumVectorOperations += dbl_assertEqual(vectorCpy[2], 27, "vectorMul (1, 2, 3) with 9");
    free(vectorCpy);
    free(vector1);
    free(vector2);
    free(vector3);

    printf("\nRunning more tests for neuralNetworkUtility.c functions\n\n");

    int testSumUtility2 = 0;

    nn = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    initNeuralNetwork(nn, 5, 1, 3, 2);
    initNeuralNetworkFunctions(nn, &sigmoid, &sigmoidDerivative, &sigmoid, &sigmoidDerivative);
    for (int i = 0; i < nn->nrOfHiddenNeurons; i++) { nn->hiddenActivationVector[i] = 1; }
    for (int i = 0; i < nn->nrOfOutputNeurons; i++) { nn->activationOutputVector[i] = 2; }
    double * activationValues = findConnectedNeuronActivations(nn, 4);
    testSumUtility2 += dbl_assertEqual(activationValues[0], 1, "findConnectedNeuronActivations 5, 1, 3, 2 network");
    testSumUtility2 += dbl_assertEqual(activationValues[1], 1, "findConnectedNeuronActivations 5, 1, 3, 2 network");
    testSumUtility2 += dbl_assertEqual(activationValues[2], 1, "findConnectedNeuronActivations 5, 1, 3, 2 network");
    testSumUtility2 += dbl_assertNotEqual(activationValues[3], 1, "findConnectedNeuronActivations 5, 1, 3, 2 network");
    activationValues = findConnectedNeuronActivations(nn, 5);
    testSumUtility2 += dbl_assertEqual(activationValues[0], 2, "findConnectedNeuronActivations 5, 1, 3, 2 network");
    testSumUtility2 += dbl_assertEqual(activationValues[1], 2, "findConnectedNeuronActivations 5, 1, 3, 2 network");
    testSumUtility2 += dbl_assertNotEqual(activationValues[2], 2, "findConnectedNeuronActivations 5, 1, 3, 2 network");
    freeNeuralNetwork(nn);

    nn = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    initNeuralNetwork(nn, 5, 1, 3, 2);
    initNeuralNetworkFunctions(nn, &sigmoid, &sigmoidDerivative, &sigmoid, &sigmoidDerivative);
    for (int i = 0; i < nn->neuronsPerLayer; i++) { nn->weightMatrix[0][i] = 1; }
    for (int i = 0; i < nn->nrOfOutputNeurons; i++) { nn->weightMatrix[7][i] = 2; }
    double const * weightValues = findOutputWeights(nn, 0);
    testSumUtility2 += dbl_assertEqual(weightValues[0], 1, "findConnectedWeights 5, 1, 3, 2 network");
    testSumUtility2 += dbl_assertEqual(weightValues[1], 1, "findConnectedWeights 5, 1, 3, 2 network");
    testSumUtility2 += dbl_assertEqual(weightValues[2], 1, "findConnectedWeights 5, 1, 3, 2 network");
    testSumUtility2 += dbl_assertNotEqual(weightValues[3], 1, "findConnectedWeights 5, 1, 3, 2 network");
    weightValues = findOutputWeights(nn, 7);
    testSumUtility2 += dbl_assertEqual(weightValues[0], 2, "findConnectedWeights 5, 1, 3, 2 network");
    testSumUtility2 += dbl_assertEqual(weightValues[1], 2, "findConnectedWeights 5, 1, 3, 2 network");
    testSumUtility2 += dbl_assertNotEqual(weightValues[2], 2, "findConnectedWeights 5, 1, 3, 2 network");
    freeNeuralNetwork(nn);

    printf("\nRunning tests for randomValueGenerator.c to check for good distribution\n\n");

    int testSumRandomValueGenerator = 0;

    setSeed(time(NULL));
    double sum = 0;
    for (int i = 0; i < 10000; i++) {
        sum += randomDouble(-10, 10);
    }
    double average = sum / 10000;
    testSumRandomValueGenerator += dbl_assertBetween(-0.2, 0.2, average, "uniformRandomValue mean -10, 10");

    sum = 0;
    double ssd = 0;
    double val = 0;
    for (int i = 0; i < 10000; i++) {
        val = boxMuellerTransform(6, 2);
        ssd += (val - 6) * (val - 6);
        sum += val;
    }
    average = sum / 10000;
    double stddev = sqrt(ssd / 10000);

    testSumRandomValueGenerator += dbl_assertBetween(5.8, 6.2, average, "boxMuellerTransform mean 6, 4");
    testSumRandomValueGenerator += dbl_assertBetween(1.8, 2.2, stddev, "boxMuellerTransform stdev 6, 4");

    if (!testSumUtility && !testSumUtility2) { printf("\nAll tests for neuralNetworkUtility.c passed\n"); } else { printf("\n\nSome tests for neuralNetworkUtility.c failed\n"); }
    if (!testSumVectorOperations) { printf("All tests for vectorOperations.c passed\n"); } else { printf("Some tests for vectorOperations.c failed\n"); }
    if (!testSumRandomValueGenerator) { printf("All tests for randomValueGenerator.c passed\n");} else { printf("Some tests for randomValueGenerator.c failed\n"); }
 
    return testSumUtility + testSumVectorOperations + testSumUtility2 + testSumRandomValueGenerator;
}