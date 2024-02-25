
#ifndef funcPtrs_h
    #define funcPtrs_h

        /** pointer for a "double func(double) --> double" function */
        typedef double (*dblA)(double);
        /** pointer for a "double func(double, double) --> double" function */
        typedef double (*dblA_dblA)(double, double);
        /** pointer for a "double func(double *, double *, int) --> double" function */
        typedef double (*dblP_dblP_intA)(double *, double*, int);
        /** pointer for a "double func(double , double, int) --> void" function */
        typedef double (*dblA_dblA_intA)(double, double, int);
#endif
