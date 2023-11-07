import pandas as pd
import numpy  as np

import tensorflow as tf
from tensorflow import keras

import pickle
import category_encoders

file="Data/datayouwannapredict.csv"              #read the file you wanna predict, must have the colnames.

targetnames=["Tidak banjir", "Banjir"]
with open("assets/onehotencoder.pkl", "br") as fh:  
    encoder = pickle.load(fh)

model=keras.models.load_model("assets/model.keras")

def predict(features):
    features=encoder.transform(features)
    probability=model.predict(features)
    predicted = tf.where(probability < 0.5, 0, 1)
    return [targetnames[0] for i in np.array(predicted)], probability.flatten()

df=pd.read_csv(file)
target, probability=predict(df)

result=df.copy()
result["predicted"]=target
result["probability"]=probability
result.to_csv("Data/Result.csv")             #location of the result