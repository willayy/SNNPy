#include "neuralNetworkFuncs.h"
#include "neuralNetworkStructs.h"

int main() {
    struct ParameterLayer parameterLayer = createParameterLayer(10);
    struct NeuronLayer neuronLayer = createNeuronLayer(3, 3);
    return 0;
}