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

    if (!testSumRandomValueGenerator) { printf("All tests for randomValueGenerator.c passed\n");} else { printf("Some tests for randomValueGenerator.c failed\n"); }
 
    return testSumRandomValueGenerator;
}