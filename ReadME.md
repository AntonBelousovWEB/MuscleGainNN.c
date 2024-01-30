# Muscle Gain NN

The Muscle Gain neural network learns and predicts weight gain based on training data.

```Xnorm = X / max(x)``` and ```Ynorm = Y / max(y)```
```Z^(2) = XW```
```A^(2) = f(Z^(2))```
```Z^(3) = A^(2)W^(2)```
```Yhat = f(Z^(3))```

- Neural Network Visualization
<img src="https://i.ibb.co/7bR5G4H/2024-01-30-204250.png">

# Description

<h4>Data Initialization:</h4>
- ```raw_x``` and ```raw_y``` arrays represent the input features and target values for training.
<br>
<h4>Weight Initialization:</h4>
- Weights for the connections between the input layer and the hidden layer are stored in the ```syn0``` array.
- Weights for the connections between the hidden layer and the output layer are stored in the ```syn1``` array.
- The ```weight_random_initialization``` and ```weight_random_initialization_1d``` functions initialize these weights randomly.
<br>
<h4>Normalization:</h4>
- The ```normalize_data_2d``` function is used to normalize the input data.
<h4>Neural Network Computation:</h4>
- The program focuses on predicting the mass (```train_y``) based on the input features (```train_x``) for a specific example (```train_x_eg1``` and ```train_y_eg1```).
- The neural network is then computed step by step:
    - ```multiple_input_multiple_output_nn```: Computes the weighted sum for each node in the hidden layer.
    - ```vector_sigmoid```: Applies the sigmoid activation function to the hidden layer nodes.
    - ```multiple_input_single_output```: Computes the weighted sum for the output layer.
    - ```sigmoid```: Applies the sigmoid activation function to the output layer
- The final output (```yhat_eg1```) represents the predicted mass.

Happy coding!