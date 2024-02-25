#include "funcPtrs.h"

#ifndef vectorOperations_h
    #define vectorOperations_h

        double dotProduct(const double * a,  const double * b, int vectorSize);

        void vectorMul(double * a, double b, int vectorSize);

        void vectorAdd(double * a, const double * b, int vectorSize);

        void vectorReplace(double * a, const double * b, int vectorSize);

        void vectorOperation(double * a,  dblA operation, int vectorSize);

        void vectorDiv(double * a, double b, int vectorSize);

        double * vectorMulCopy(const double * a, double b, int vectorSize);

        double * vectorCopy(const double * a, int vectorSize);

        void vectorExtend(double * a, const double * b, int from, int vectorSize);
#endif