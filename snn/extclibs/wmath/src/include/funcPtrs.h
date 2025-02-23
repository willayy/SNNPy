#ifndef funcPtrs_h

    #define funcPtrs_h

        /** pointer for a "double func(double)" function */
        typedef double (*ActivationFunc)(double);

        /** pointer for a "double func(double)" function */
        typedef double (*d_ActivationFunc)(double);

        /** pointer for a "double func(double *, double *, int)" function"*/
        typedef double (*CostFunc)(double *, double *, int);

        /** pointer for a "double func(double, double)" function */
        typedef double (*d_CostFunc)(double, double);
        
#endif
