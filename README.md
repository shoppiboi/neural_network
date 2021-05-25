# Neural Network

## About
A basic OOP implementation of a neural network for the recognition of handwritten digits.

### A Note Regarding the Convolutional and Pooling layer scripts:
*They were created while I was exploring how convolutional and pooling layers could be coded.*

*Unfortunately I didn't find the time to complete the backpropagation for the layers, but I didn't want to lose the work I had done so far, as I intend to finish this project sometime over the summer, hence I decided to include the code in this repository as well.*

*(I also thought these were pretty cool and the way I managed to capture the essence of forward propagation for these layers is something I'm very proud of.)*

**Libraries Used**:
- [NumPy](https://numpy.org/)

**Resources used:**
* [*Make Your Own Neural Network*, Tariq Rashid](https://www.amazon.co.uk/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G)
* [*CS231n Convolutional Neural Networks for Visual Recognition*, Stanford](https://cs231n.github.io/convolutional-networks/)
* [*Backpropagation in Convolutional Neural Networks*, Jefkine](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)

## Results
### Fully Connected Network
    +-------------------------+
    | Fully Connected Network |
    +-------------------------+
    Learning Rate: 0.1
    Input Shape: (784,)
    +-----------------------+--------------+
    | Layer Type            | Output Shape |
    +-----------------------+--------------+
    | Fully Connected Layer | (200,)       |
    | Activation (Sigmoid)  | (200,)       |
    | Fully Connected Layer | (10,)        |
    | Activation (Sigmoid)  | (10,)        |
    +----------------------+---------------+
    
    Accuracy: 0.9732
