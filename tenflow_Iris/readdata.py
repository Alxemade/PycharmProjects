import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

data = pd.read_csv("iris_training.csv",names=CSV_COLUMN_NAMES,header=0)
train_x, train_y = data, data.pop('Species')
print(train_y)

