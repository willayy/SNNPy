from snnpy import snn_py

inputs = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]]
outputs = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]
snn_py.set_rng_seed(0)
neural_network = snn_py.create_neural_network(4,1,4,16)
snn_py.set_input_layer_activation_function(neural_network, 'linear')
snn_py.set_hidden_layer_activation_function(neural_network, 'relu')
snn_py.set_output_layer_activation_function(neural_network, 'sigmoid')
snn_py.init_weights_xavier_normal(neural_network)
snn_py.init_biases_constant(neural_network, 0.1)
snn_py.train_neural_network(neural_network, inputs, outputs, 16, 250000, 0.08, 0.01, 'cross_entropy')