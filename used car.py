import os
import time
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.model_selection as sk

# DATASET_PATH = os.path.join(os.getcwd(), "g", "g_dataset")
DATASET_PATH = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'usedcp')
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train-data.csv')
TEST_DATASET_PATH = os.path.join(DATASET_PATH, 'test-data.csv')


def get_preprocessing_csv(csv_path):
    csv_data = pd.read_csv(csv_path)
    csv_data = csv_data.drop(['Unnamed: 0', 'New_Price'],
                             axis=1)  # Null값이 많은 New_Price 열 삭제
    csv_data = csv_data.dropna()  # null 값 및 NaN 값 제거
    csv_data = csv_data.mask(csv_data.eq('None')).dropna()  # 문자열로 None인 경우 제거
    csv_data = csv_data.mask(csv_data.eq('null')).dropna()  # 문자열로 null인 경우 제거

    # 브랜드 이름 통일, Mileage의 km/kg제거 Engine의 CC제거, Power의 bhp제거
    for col in ['Name', 'Mileage', 'Engine', 'Power']:
        csv_data[col] = csv_data[col].map(lambda x: x.split(
            ' ', 1)[0] if type(x) == str else None)

    # str 타입을 float 타입으로 변환
    for col in ["Mileage", "Engine", "Power"]:
        csv_data[col] = csv_data[col].map(
            lambda x: float(x) if x != 'null' else None)
    csv_data = csv_data.dropna()  # null 값 및 NaN 값 제거

    # 카테고리컬로 분류하기 Name, Location, Owner_Type, Seats
    for col in ['Year', 'Seats']:
        csv_data[col] = pd.Categorical(csv_data[col])
    csv_data = pd.get_dummies(csv_data, prefix_sep='_', drop_first=True)
    y_data = csv_data['Price']
    csv_data = csv_data.drop(['Price'], axis=1)
    return csv_data.to_numpy(), y_data.to_numpy()

def get_LSTM_model():
  model = tf.keras.Sequential([
      tf.keras.layers.LSTM(units=78, activation='relu', input_shape=(1, 78), return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001)),
      tf.keras.layers.LSTM(units=78, activation='relu', return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1, activation='relu')
  ])
  return model
  
def get_GRU_model():
  model = tf.keras.Sequential([
      tf.keras.layers.GRU(units=78, activation='relu', input_shape=(1, 78), return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001), bias_regularizer=tf.keras.regularizers.l2(0.001)),
      tf.keras.layers.GRU(units=78, activation='relu', return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1, activation='relu')
  ])
  return model

def simulation(mode, epochs, model):
  model_path = os.path.join(os.getcwd(), 'drive', 'MyDrive','car_model', mode)
  logs_path = os.path.join(os.getcwd(), 'drive', 'MyDrive','car_logs', mode)
  x_full_data, y_full_data = get_preprocessing_csv(TRAIN_DATASET_PATH)

  x_data, x_test, y_data, y_test = sk.train_test_split(
      x_full_data, y_full_data, test_size=0.1, shuffle=True)
  x_train, x_valid, y_train, y_valid = sk.train_test_split(
      x_data, y_data, test_size=0.2, shuffle=True)

  x_train = np.array(x_train).reshape(x_train.shape[0],1, x_train.shape[1])
  x_valid = np.array(x_valid).reshape(x_valid.shape[0],1, x_valid.shape[1])
  x_test = np.array(x_test).reshape(x_test.shape[0],1, x_test.shape[1])

  # model.summary()
  model.compile(loss='mse', optimizer='adam', metrics=['mae'])
  tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path)
  model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, batch_size=16, verbose=2, callbacks=[tensorboard])
  model.save(model_path)
  score = model.evaluate(x_test, y_test)
  print(score)

simulation('lstm100', 100, get_LSTM_model())
simulation('lstm1000', 1000, get_LSTM_model())
simulation('gru1000', 1000, get_GRU_model())