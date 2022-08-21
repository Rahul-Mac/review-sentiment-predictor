"""
This file is part of Review Sentiment Predictor.

Review Sentiment Predictor is free software: you can redistribute it and/or modify it under the terms of the GNU General
Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

Review Sentiment Predictor is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Review Sentiment Predictor. If not, see
<https://www.gnu.org/licenses/>.
"""

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
from keras.models import load_model

def predict_sentiment(text):
    t = token.texts_to_sequences([text])
    t = pad_sequences(t, maxlen = 200)
    pred = int(model.predict(t).round().item())
    print("Predicted label: ", label[1][pred])

df = pd.read_csv("data.csv")
label = df.sentiment.factorize()
text = df.review.values
token = Tokenizer(num_words = 5000)
token.fit_on_texts(text)
vocab_size = len(token.word_index) + 1
encoded_docs = token.texts_to_sequences(text)
pad_seq = pad_sequences(encoded_docs, maxlen = 200)
embedding_vector_length = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length = 200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout = 0.5, recurrent_dropout = 0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
model.fit(pad_seq, label[0], validation_split = 0.2, epochs = 1, batch_size = 32)

model.save('model.h5')
del model
model = load_model('model.h5')

while(True):
    sample = input("Write a review: ")
    predict_sentiment(sample)
