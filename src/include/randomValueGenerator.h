#ifndef randomValueGenerator_h
    #define randomValueGenerator_h

        int getSeed();

        void setSeed(unsigned int seed);

        double randomValue(double min, double max);

        double boxMuellerTransform(double mean, double stddev);
#endif