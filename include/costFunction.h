#ifndef costFunction_h
    #define costFunction_h

    double costFunctionDerivative(double output, double desiredOutput);

    double costFunction(double * output, double * desiredOutput, int outputSize);
    
#endif