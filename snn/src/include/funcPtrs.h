

#ifndef funcPtrs_h
    #define funcPtrs_h

        /** pointer for a "double func(double)" function */
        typedef double (*dblA_dblR)(double);
        /** pointer for a "double func(double, double)" function */
        typedef double (*dblA_dbLA_dblR)(double, double);
        /** pointer for a "double func(double *, double *, int)" function"*/
        typedef double (*dblpA_dblpA_intA_dblR)(double *, double *, int);
#endif
