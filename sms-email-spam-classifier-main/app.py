# from flask import Flask, request, render_template, jsonify
# import pickle
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# # Initialize Flask app
# app = Flask(__name__)

# # Initialize NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# # Function to transform text
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# # Load the pre-trained vectorizer and model
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# # Define the main route
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Define the prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     input_sms = request.form['input_sms']
    
#     # Preprocess the input
#     transformed_sms = transform_text(input_sms)
    
#     # Vectorize the input
#     vector_input = tfidf.transform([transformed_sms])
    
#     # Predict the result
#     result = model.predict(vector_input)[0]
    
#     # Return the result as JSON
#     return jsonify(result="Spam" if result == 1 else "Not Spam")

# if __name__ == '__main__':
#     app.run(debug=True,port=5001)

from flask import Flask, request, render_template, jsonify
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the pre-trained pipeline
pipeline = pickle.load(open('spam_classifier.pkl', 'rb'))

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form['input_sms']
    
    # Preprocess the input
    transformed_sms = transform_text(input_sms)
    
    # Predict the result using the pipeline
    result = pipeline.predict([transformed_sms])[0]
    
    # Return the result as JSON
    return jsonify(result="Spam" if result == 1 else "Not Spam")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
