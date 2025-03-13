## Stock Price Prediction using LSTM and MLP

This repository contains PyTorch implementations for stock price prediction using Long Short-Term Memory (LSTM) networks and Multi-Layer Perceptron (MLP) models.

### Models

The project implements four main approaches:

- **LSTM Regression**: Time-series prediction of stock prices
- **LSTM Classification**: Predicting price movement direction (up/down)
- **MLP Regression**: Feed-forward network for price prediction
- **MLP Classification**: Feed-forward network for direction classification

### Model Architectures

#### LSTM Models

- Three LSTM layers with dropout for regularization
- Hidden dimensions of 64 and 32 units
- Tanh activation for regression, Sigmoid for classification

#### MLP Models

- Four fully connected layers with ReLU activations
- Hidden layer sizes of [64, 32, 16]
- Dropout layers for regularization in classification models

### Data Preprocessing

- Time series conversion using the `series_to_supervised` function
- Data reshaping to fit model input requirements [samples, timesteps, features]
- Data normalization

### Training

- Adam optimizer with learning rate of 0.001
- MSE loss for regression tasks
- BCE loss for classification tasks
- Training with validation monitoring

### Visualization

Results are visualized using the `plot_results` function to compare actual vs predicted values.

### Implementation

The models are implemented in Jupyter notebooks:

- `LSTM_prediction(regression).ipynb`
- `LSTM_classification.ipynb`
- `MLP_prediction(regression).ipynb`
- `MLP_Classification.ipynb`
