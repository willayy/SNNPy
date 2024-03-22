#include "neuralNetworkStructs.h"
#include "neuralNetworkUtility.h"
#include "neuralNetworkInit.h"
#include "randomValueGenerator.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "testing.h"
#include "activationFunctions.h"
#include "nnMemManagement.h"
#include "costFunctions.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

int runTests() {

    ///////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////      TESTING COST FUNCTIONS       //////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    
    printf("\nRunning tests for cost function \n\n");

    int testSumCostFunction = 0;

    double desiredOutput[] = {1, 0, 0, 0, 0};
    double output[] = {1, 0, 0, 0, 0};
    testSumCostFunction = dbl_assertEqual(0, crossEntropyCostFunction(output, desiredOutput, 5), "crossEntropyCost function cost");

    if (!testSumCostFunction) { printf("All tests for cost functions passed\n");} else { printf("Some tests for cost functions failed\n"); }

    ///////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////      TESTING RANDOM VALUE GENERATOR       //////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////

    printf("\nRunning tests for random value generator to check for good distribution\n\n");

    int testSumRandomValueGenerator = 0;

    setRngSeed(time(NULL));
    double sum = 0;
    for (int i = 0; i < 10000; i++) {
        sum += randomDouble(-10, 10);
    }
    double average = sum / 10000;
    testSumRandomValueGenerator += dbl_assertBetween(-0.2, 0.2, average, "uniformRandomValue mean");

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

    testSumRandomValueGenerator += dbl_assertBetween(5.8, 6.2, average, "boxMuellerTransform mean");
    testSumRandomValueGenerator += dbl_assertBetween(1.8, 2.2, stddev, "boxMuellerTransform stdev");

    

    if (!testSumRandomValueGenerator) { printf("All tests for random value generator passed\n");} else { printf("Some tests for random value generator failed\n"); }

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //////////////////////////      TESTING FORWARD AND BACK PROPOGATION       //////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

    printf("\nRunning tests for forward and back propogation\n\n");

    int testSumForwardBackPropogation = 0;

    // Setting up a test neural network
    NeuralNetwork * nn  = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));
    initNeuralNetwork(nn, 2, 1, 3, 2);
    setInputLayerActivationFunction(nn, &linear, &linearDerivative);
    setHiddenLayerActivationFunction(nn, &rectifiedLinearUnit, &rectifiedLinearUnitDerivative);
    setOutputLayerActivationFunction(nn, &sigmoid, &sigmoidDerivative);
    initBiasesConstant(nn, 0.1);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            nn->neurons[i]->weights[j] = 0.5;
        }
    }

    for (int i = 2; i < 5; i++) {
        for (int j = 0; j < 2; j++) {
            nn->neurons[i]->weights[j] = 0.5;
        }
    }

    double input[] = {1, 0.7};
    double desOutput[] = {0.7, 1};

    double * nnoutput = inputDataToNeuralNetwork(nn, input);

    testSumRandomValueGenerator += dbl_assertBetween(0.81, 0.83, nnoutput[0], "forward propogation output 1");
    testSumRandomValueGenerator += dbl_assertBetween(0.82, 0.84, nnoutput[1], "forward propogation output 2");

    GradientVector * gv = computeGradients(nn, desOutput, &sqrCostFunctionDerivative);

    // TODO: Control that all gradient values are the same as manually calculated values.
    
    freeGradientVector(gv);
    free(nnoutput);
    freeNeuralNetwork(nn);

    if (!testSumForwardBackPropogation) { printf("All tests for forward and back propogation passed\n\n");} else { printf("Some tests for forward and back propogation failed\n\n"); }

    return testSumRandomValueGenerator + testSumCostFunction + testSumForwardBackPropogation;
}