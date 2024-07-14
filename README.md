# data-science
Some notebooks and datasets for data science training. Mostly about stocks.

You can find more in the folder. By training the neural network in tensorflow.ipynb longer (ie more than 8000 epochs) you might be able to get a better rmse value. But I don't have the time or computational resources to do that. :)

You can use the model I managed to train by loading the ```AAPL8000.keras``` model using ```load_model()```:
```
new_model = tf.keras.models.load_model('AAPL8000.keras')

# Show the model architecture
new_model.summary()
```

Other notebooks were using SK Learn.
