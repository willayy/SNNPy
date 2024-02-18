#ifndef vectorOperations_h
    #define vectorOperations_h

        typedef double (*singeleDoubleParamOperation)(double);

        double dotProduct(double *a, double *b, int vectorSize);

        void vectorMul(double * a, double b, int vectorSize);

        void vectorAdd(double * a, double * b, int vectorSize);

        void vectorReplace(double * a, double * b, int vectorSize);

        void vectorOperation(double * a, singeleDoubleParamOperation operation, int vectorSize);

        double * vectorMulCopy(double * a, double b, int vectorSize);

        double * vectorCopy(double * a, int vectorSize);

        void vectorExtend(double * a, double * b, int from, int vectorSize);
#endif