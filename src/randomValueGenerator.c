#include "randomValueGenerator.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

static double pi = 3.14159265358979323846;

static int seed = 0;

int getSeed() {
    return seed;
}

void setSeed(unsigned int newSeed) {
    seed = newSeed;
    srand(seed);
}

/**
 * Generates a random value between min and max uniformly.
 * @param min: the lower bound of the random value.
 * @param max: the upper bound of the random value.
 * @return a random value between min and max. */
double randomValue(int min, int max) {
    return ( (double)rand() * ( max - min ) ) / (double)RAND_MAX + min;
}

/**
 * Generates a normally distributed random value using the Box-Mueller transform.
 * @param mean: the mean of the normal distribution.
 * @param stddev: the standard deviation of the normal distribution.
 * @return a random value from a normal distribution with the given mean and standard deviation. */
double boxMuellerTransform(double mean, double stddev) {
    double u1 = rand() / (RAND_MAX + 1.0);
    double u2 = rand() / (RAND_MAX + 1.0);
    
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
    
    return mean + z0 * stddev;
}