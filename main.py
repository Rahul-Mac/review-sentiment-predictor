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

from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

class Window:
    def __init__(self):
        self.window = Tk()
        self.window.title("Review Sentiment Predictor")
        self.window.resizable(0,0)
        self.window.geometry("780x580")
        self.window.iconbitmap("icon.ico")
        self.widgets()
        self.flag = 0
        self.window.mainloop()

    def widgets(self):
        self.frame = ttk.LabelFrame(self.window, text = "Review")
        self.frame.grid(column = 0, row = 0, padx = 5, pady = 5)

        self.text = Text(self.frame)
        self.text.grid(column = 0, row = 0, padx = 5, pady = 5)
        self.text.configure(font = ("Times New Roman", 14))
        self.scroll = ttk.Scrollbar(self.frame, orient = 'vertical', command = self.text.yview)
        self.scroll.grid(row = 0, column = 1, sticky = NS, padx = 5, pady = 5)
        self.text['yscrollcommand'] = self.scroll.set

        self.btn = ttk.Button(self.frame, text = "Check Sentiment", command = self.check)
        self.btn.grid(column = 0, row = 1, padx = 5, pady = 5)

    def check(self):
        text = self.text.get("1.0", 'end-1c')
        text = " ".join(text.split())
        if text == "" or text.isspace():
            self.error()
        else:
            if self.flag == 0:
                self.model()
            self.prediction(text)

    def message(self, x):
        messagebox.showinfo("Message", "Predicted Sentiment: " + x.upper())
  
    def error(self):
        messagebox.showerror("Message", "Review box cannot be empty")

    def model(self):
        self.df = pd.read_csv("data.csv")
        self.label = self.df.sentiment.factorize()
        self.review = self.df.review.values
        self.token = Tokenizer(num_words = 5000)
        self.token.fit_on_texts(self.review)
        self.m = load_model('model.h5')
        self.flag = 1

    def prediction(self, text):
        t = self.token.texts_to_sequences([text])
        t = pad_sequences(t, maxlen = 200)
        pred = int(self.m.predict(t).round().item())
        self.message(self.label[1][pred])

if __name__ == "__main__":
    Window()
