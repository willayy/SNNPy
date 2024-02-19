#include "randomValueGenerator.h"
#include <stdlib.h>
#include <time.h>

static int seed = 0;

int getSeed() {
    return seed;
}

void setSeed(unsigned int newSeed) {
    seed = newSeed;
    srand(seed);
}

double randomValue(int min, int max) {
    return ( (double)rand() * ( max - min ) ) / (double)RAND_MAX + min;
}