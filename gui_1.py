import tkinter as tk
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and the TF-IDF vectorizer from the pickle files
with open('email_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('email_vectorizer.pkl', 'rb') as f:
    feature_extraction = pickle.load(f)


def check_email():
    # Get the input text from the GUI
    input_text = text_entry.get()

    # Vectorize the input text using the TF-IDF vectorizer
    input_data_features = feature_extraction.transform([input_text])

    # Use the trained model to predict whether the input text is spam or not
    prediction = clf.predict(input_data_features)

    # Display the prediction in the GUI
    result_label.config(text=f'Prediction: {"Spam" if prediction[0] == 0 else "Not Spam"}')

# Create the main window of the GUI
root = tk.Tk()

# Create a label and an entry widget for the input text
input_label = tk.Label(root, text='Enter email text:')
input_label.pack()
text_entry = tk.Entry(root, width=50)
text_entry.pack()

# Create a button to check the input text
check_button = tk.Button(root, text='Check', command=check_email)
check_button.pack()

# Create a label to display the prediction
result_label = tk.Label(root, text='')
result_label.pack()

# Run the Tkinter event loop
root.mainloop()