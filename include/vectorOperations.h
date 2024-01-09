#ifndef vectorOperations_h
#define vectorOperations_h

double dotProduct(double *a, double *b, int vectorSize);

double * vectorMul(double * a, double b, int vectorSize);

void vectorAdd(double * a, double * b, int vectorSize);

void vectorReplace(double * a, double * b, int vectorSize);

#endif