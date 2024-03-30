#include <stdlib.h>
#include <math.h>
#include "neuralNetworkStructs.h"
#include "neuralNetworkInit.h"
#include "randomValueGenerator.h"
#include "neuralNetworkOperations.h"
#include "neuralNetworkTraining.h"
#include "testing.h"
#include "activationFunctions.h"
#include "nnMemManagement.h"
#include "costFunctions.h"


int runTests() {

    ///////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////      TESTING COST FUNCTIONS       //////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////////////////////////////////

    int testSumCostFunction = 0;

    double desiredOutput[] = {1, 0, 0, 0, 0};
    double output[] = {1, 0, 0, 0, 0};
    testSumCostFunction = dbl_assertEqual(0, crossEntropyCostFunction(output, desiredOutput, 5), "");

    ///////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////      TESTING RANDOM VALUE GENERATOR       //////////////////////////

    ///////////////////////////////////////////////////////////////////////////////////////////////

    int testSumRandomValueGenerator = 0;

    setRngSeed(0);
    double sum = 0;
    for (int i = 0; i < 10000; i++) {
        sum += randomDouble(-10, 10);
    }
    double average = sum / 10000;
    testSumRandomValueGenerator += dbl_assertBetween(-0.2, 0.2, average, "");

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

    testSumRandomValueGenerator += dbl_assertBetween(5.8, 6.2, average, "");
    testSumRandomValueGenerator += dbl_assertBetween(1.8, 2.2, stddev, "");

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //////////////////////////      TESTING FORWARD AND BACK PROPOGATION       //////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////

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

    testSumForwardBackPropogation += dbl_assertBetween(0.80, 0.85, nnoutput[0], "");
    testSumForwardBackPropogation += dbl_assertBetween(0.80, 0.85, nnoutput[1], "");
    
    free(nnoutput);
    freeNeuralNetwork(nn);

    return testSumRandomValueGenerator + testSumCostFunction + testSumForwardBackPropogation;
}