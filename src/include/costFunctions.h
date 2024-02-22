#ifndef costFunctions_h
    #define costFunctions_h

        double sqrCostFunctionDerivative(double output, double desiredOutput);

        double sqrCostFunction(double * output, double * desiredOutput, int outputSize);

        double * elementWiseSqrCost(double * output, double * desiredOutput, int outputSize);

        double crossEntropyCostFunctionDerivative(double output, double desiredOutput);

        double crossEntropyCostFunction(double * output, double * desiredOutput, int outputSize);

        double * elementWiseCrossEntropyCost(double * output, double * desiredOutput, int outputSize);
#endif