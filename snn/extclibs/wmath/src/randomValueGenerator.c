#include <stdlib.h>
#include <math.h>
#include "randomValueGenerator.h"

static double pi = 3.14159265358979323846;

static int seed = 0;

int getRngSeed() {
    return seed;
}

void setRngSeed(unsigned int newSeed) {
    seed = newSeed;
    srand(seed);
}

/**
 * Generates a random value between min and max (inclusive) uniformly.
 * @param min: the lower bound of the random value.
 * @param max: the upper bound of the random value.
 * @return a random value between min and max. */
double randomDouble(double min, double max) {
    return ( (double) rand() * ( max - min ) ) / (double)RAND_MAX + min;
}

/**
 * Generates a random integer between min and max (inclusive) uniformly.
 * @param min: the lower bound of the random integer.
 * @param max: the upper bound of the random integer.
 * @return a random integer between min and max. */
int randomInt(int min, int max) {
    return rand() % (max - min + 1) + min;
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

/**
 * Shuffles an array using the Fisher-Yates algorithm.
 * @param arr: the array to shuffle.
 * @param n: the length of the array. */
void fisherYatesShuffle(int * arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        // Generate a random index between 0 and i
        int j = rand() % (i + 1);
        
        // Swap arr[i] with arr[j]
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}