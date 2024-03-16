#ifndef randomValueGenerator_h
    #define randomValueGenerator_h

        int getSeed();

        void setSeed(unsigned int seed);

        double randomDouble(double min, double max);

        int randomInt(int min, int max);

        double boxMuellerTransform(double mean, double stddev);

        void fisherYatesShuffle(int * arr, int n);
#endif