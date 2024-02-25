#ifndef costFunctions_h
    #define costFunctions_h

        double sqrCostFunctionDerivative(double output, double desiredOutput, int batchSize);

        double sqrCostFunction(double * output, double * desiredOutput, int outputSize);

        double crossEntropyCostFunctionDerivative(double output, double desiredOutput, int batchSize);

        double crossEntropyCostFunction(double * output, double * desiredOutput, int outputSize);
#endif