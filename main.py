import numpy as np
import layerlibrary as ll
import activationlibrary as ac

LEARNING_RATE = 0.3

class FullyConnectedLayer:
    
    def __init__(self, inputs, outputs):

        self.input_nodes = inputs
        self.output_nodes = outputs

        #   "He initialization" for the weights
        self.weights = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.output_nodes, self.input_nodes))

    def forward_propagation(self, inputs):

        dot_product = np.dot(self.weights, inputs)

        return np.array(dot_product, ndmin=2)

    def back_propagation(self, error_derivative):
        
        self.weights += LEARNING_RATE * error_derivative

def main():

    layers = {
        "FC1" : ll.FullyConnectedLayer(784, 200),
        "FC2" : ll.FullyConnectedLayer(200, 10)
    }

    training_data_file = open(r"mnist_train.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 5

    #   training phase
    for e in range(epochs):
        for record in range(1, len(training_data_list)):
            all_values = training_data_list[record].split(',')

            #   simplify all values between 0.01 and 0.99 to keep the images black and white
            inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
            inputs = np.array(inputs, ndmin=2).T
            
            targets = np.zeros(10) + 0.01
            targets[int(all_values[0])] = 0.99
            targets = np.array(targets, ndmin=2).T

            #   forward propagation through the first layer
            fc1_output = layers["FC1"].forward_propagation(inputs)
            fc1_activated = ac.sigmoid(fc1_output)

            #   forward propagation through the second layer 
            fc2_output = layers["FC2"].forward_propagation(fc1_activated)
            fc2_activated = ac.sigmoid(fc2_output)

            #   determine the errors
            errors = targets - fc2_activated
            second_errors = np.dot(layers["FC2"].weights.T, errors)

            #   backpropagation through second layer
            fc2_errors = np.dot((errors * ac.sigmoid_derivative(fc2_activated)), np.transpose(fc1_activated))
            layers["FC2"].back_propagation(fc2_errors)

            #   backpropagation through first layer
            fc1_errors = np.dot((second_errors * ac.sigmoid_derivative(fc1_activated)), np.transpose(inputs))
            layers["FC1"].back_propagation(fc1_errors)
            pass
    pass

    test_data_file = open(r"mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    performance = []

    #   testing phase
    #   exact same setup as training phase, except without epochs.
    for record in range(1, len(test_data_list)):
        all_values = test_data_list[record].split(',')
        correct_label = int(all_values[0])

        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        inputs = np.array(inputs, ndmin=2).T

        fc1_output = layers["FC1"].forward_propagation(inputs)
        fc1_activated = ac.sigmoid(fc1_output)

        fc2_output = layers["FC2"].forward_propagation(fc1_activated)
        fc2_activated = ac.sigmoid(fc2_output)

        label = np.argmax(fc2_activated)
        print("c: " + str(correct_label) + " and l: " + str(label))

        if (label == correct_label):
            performance.append(1)
        else:
            performance.append(0)

    score = np.asarray(performance)
    print("network performance: ", score.sum() / score.size)


if __name__ == '__main__':
    main()