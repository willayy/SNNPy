#include <stdio.h>
#include <stdlib.h>

struct Neuron
{
    double* weights; // The weights of the edges coming into this neuron
    double output; // The output of this neuron
};

/**
 * Initialize a starting vector of neurons
 * @param num_inputs The number of parameters
 * @return The initialized parameter layer pointer
*/
struct Neuron* makeParameterLayer(int num_inputs) {
    struct Neuron* parameterLayer = malloc(sizeof(struct Neuron) * num_inputs);
    for (int i = 0; i < num_inputs; i++)
    {
        parameterLayer[i] = initializeNeuron(0);
    }
    return parameterLayer;
}

/**
 * Initialize a neuron with weights set to 1
 * @param num_inputs The number of inputs to this neuron
 * @return The initialized neuron
*/
struct Neuron initializeNeuron(int num_inputs) {
    struct Neuron neuron;
    neuron.weights = malloc(sizeof(double) * num_inputs);
    for (int i = 0; i < num_inputs; i++)
    {
        neuron.weights[i] = 1;
    }
    return neuron;
}

/**
 * Free the memory allocated to a neuron
 * @param neuron The neuron to free from memory
*/
void freeNeuron(struct Neuron *neuron) {
    free(neuron->weights);
}