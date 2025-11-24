# Email-Spam-Detection-
Email Spam Detection 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/Project/SPAM/spam (1).csv")

# Remove duplicates
data.drop_duplicates(inplace=True)

# Replace categories
data['Category'] = data['Category'].replace(['ham', 'spam'], ['not spam', 'spam'])

# Split data
X = data['Message']
y = data['Category']

mess_train, mess_test, cat_train, cat_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numerical features
cv = CountVectorizer(stop_words='english')
features_train = cv.fit_transform(mess_train)

# Train model
model = MultinomialNB()
model.fit(features_train, cat_train)

# Test model
features_test = cv.transform(mess_test)
predictions = model.predict(features_test)

# Model accuracy
acc = accuracy_score(cat_test, predictions)
print("Accuracy:", acc)
print("Score:", model.score(features_test, cat_test))

# Prediction function
def predict(message):
    input_message = cv.transform([message])
    result = model.predict(input_message)
    return result

# Streamlit UI
st.header('📩 Spam Detection App')
input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
    output = predict(input_mess)
    st.markdown(f"Prediction: **{output[0]}**")
