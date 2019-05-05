# Convolution_LSTM_pytorch

A multi-layer convolution LSTM  module

Pytorch implementation of  [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)


# Usage
```python
clstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64], kernel_size=5, step=9, effective_step=[2, 4, 8])
lstm_outputs = clstm(cnn_features)
hidden_states = lstm_outputs[0]
```

# UPDATE-May.18.2018
Fix a critial bug. Thanks to [@Jackie-Chou](https://github.com/Jackie-Chou) and [@chencodeX](https://github.com/chencodeX).

# UPDATE-May.04.2019
As suggested by multiple people, Wci, Wcf, and Wco should not be initialized frequently.
Apologies for the inconvenience.