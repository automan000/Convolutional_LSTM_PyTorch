# Convolution_LSTM_pytorch

Thanks for your attention. 
I haven't got time to maintain this repo for a long time.
I recommend this [repo](https://github.com/Hzzone/Precipitation-Nowcasting) which provides an excellent implementation.

# Usage
A multi-layer convolution LSTM module
Pytorch implementation of  [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)


```python
clstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64], kernel_size=5, step=9, effective_step=[2, 4, 8])
lstm_outputs = clstm(cnn_features)
hidden_states = lstm_outputs[0]
```

# Thanks
Thanks to [@Jackie-Chou](https://github.com/Jackie-Chou) and [@chencodeX](https://github.com/chencodeX) who provide lots of valuable advice.
I apology for the inconvenience.