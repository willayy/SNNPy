#ifndef randomValueGenerator_h
    #define randomValueGenerator_h

        int getSeed();

        void setSeed(unsigned int seed);

        double randomValue(double min, double max);

        int randomInt(int min, int max);

        double boxMuellerTransform(double mean, double stddev);
#endif