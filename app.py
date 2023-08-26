import streamlit as st
import pandas as pd
import os
import string
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from collections import Counter
from io import BytesIO
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import emoji

nltk.download('punkt')
nltk.download('stopwords')


st.set_option('deprecation.showPyplotGlobalUse', False)

topik_list = ["Prabowo", "Sandiuno", "Airlanggahartarto"]

# Load CSV data based on selected topic
def load_data(topic):
    file_path = os.path.join("dataset", f"{topic}.csv")
    data = pd.read_csv(file_path)
    return data

# Load positive, negative words, and stopwords
def load_positive_words():
    with open("positive.txt", "r") as file:
        positive_words = [line.strip() for line in file]
    return positive_words

def load_negative_words():
    with open("negative.txt", "r") as file:
        negative_words = [line.strip() for line in file]
    return negative_words

def load_stopwords():
    with open("stopwords-id.txt", "r") as file:
        stopwords = [line.strip() for line in file]
    return stopwords


def analyze_sentiment(text, stopwords=None):  # Provide a default value for 'stopwords'
    cleaned_text = clean_text(text, stopwords)
    sentiment_score = TextBlob(cleaned_text).sentiment.polarity
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment


def clean_text(text, stopwords=None):
    # Remove emojis from the text using regular expression
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords]
    cleaned_text = ' '.join(words)
    return cleaned_text


# Feature Extraction using TF-IDF
def feature_extraction(data):
    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(data['cleaned_text'])
    return features

# Feature Selection using SelectKBest
def feature_selection(X, y):
    selector = SelectKBest(chi2, k=100)
    X_new = selector.fit_transform(X, y)
    return X_new



def main():
    
    menu = ["Home", "Dataset", "Processing", "Visualisasi", "Model & Prediksi","About"]
    choice = st.sidebar.selectbox("Menu", menu)
    # Load the Instagram logo image
    instagram_logo = Image.open("logoig.png")

    st.sidebar.image(instagram_logo, use_column_width=True)
    st.sidebar.caption(':copyright: Sorongdev' )

    if choice == "Home":
        st.markdown("<h1 style='text-align: center; color: white;'>Aplikasi Analisis Sentimen Instagram</h1>", unsafe_allow_html=True)
       
        st.image("https://www.travelmediagroup.com/wp-content/uploads/2022/04/bigstock-Market-Sentiment-Fear-And-Gre-451706057-2880x1800.jpg",use_column_width=True)

    elif choice == "Dataset":
        st.write("Pilih Topik")
        selected_topic = st.selectbox("", topik_list)
        st.write(f"Menampilkan data untuk topik: {selected_topic}")
        # Load data for the selected topic
        data = load_data(selected_topic)
        # Display the data table
        st.dataframe(data)
        # Display the number of data
        num_data = len(data)
        st.write(f"Jumlah data: {num_data}")

    elif choice == "Processing":
        st.title("Pilih Topik")
        selected_topic = st.selectbox("", topik_list)
        st.write(f"Menampilkan data untuk topik: {selected_topic}")
        data = load_data(selected_topic)

        # Load stopwords, positive words, and negative words
        stopwords = load_stopwords()
        positive_words = load_positive_words()
        negative_words = load_negative_words()

        # Drop specified columns
        columns_to_drop = ["profilePictureUrl", "profileUrl", "likeCount", "replyCount", "commentDate", "timestamp", "commentId", "ownerId", "query"]
        data = data.drop(columns=columns_to_drop)

        data['cleaned_text'] = data['comment'].apply(clean_text, args=(stopwords,))
        data['sentiment_label'] = data['cleaned_text'].apply(analyze_sentiment, stopwords=stopwords)

        # Feature Extraction
        X = feature_extraction(data)
        y = data['sentiment_label']

        # Feature Selection
        X_selected = feature_selection(X, y)

        # Display processed data
        st.write("Data setelah preprocessing, feature extraction, dan feature selection:")
        st.write(data)
        # st.write("Data features setelah selection:")
        # st.write(X_selected)

        # Create a BytesIO object to store the CSV data
        csv_data = data.to_csv(index=False).encode()

        # Add a download button for preprocessed data
        st.download_button(
            label="Download Preprocessed Data",
            data=csv_data,
            file_name="preprocessed_data.csv",
            mime="text/csv"
        )



    elif choice == "Visualisasi":
        # st.sidebar.title("Pilih Topik")
        # selected_topic = st.sidebar.selectbox("", topik_list)
        # st.sidebar.write(f"Menampilkan data untuk topik: {selected_topic}")
        data = st.file_uploader("Upload Preprocessed Data", type=["csv"])

        if data is not None:
            data = pd.read_csv(data)

            # Handle NaN or null values in cleaned_text
            data['cleaned_text'].fillna('', inplace=True)
            # Display wordcloud
            st.write("Wordcloud:")
            wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(" ".join(data['cleaned_text'].astype(str)))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot()

            # Generate sentiment-specific word clouds
            for sentiment_label in ["Positive", "Negative", "Neutral"]:
                st.write(f"{sentiment_label} Wordcloud:")
                filtered_text = " ".join(data[data['sentiment_label'] == sentiment_label]['cleaned_text'].astype(str))
                wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(filtered_text)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot()

        # Word frequency analysis
                words = re.findall(r'\b\w+\b', filtered_text)  # Tokenize words
                word_counter = Counter(words)
                common_words = word_counter.most_common(10)

                # Create a bar chart for the most common words
                word_labels, word_counts = zip(*common_words)
                plt.figure(figsize=(10, 6))
                plt.bar(word_labels, word_counts)
                plt.xlabel("Words")
                plt.ylabel("Frequency")
                plt.title(f"Most Common Words for {sentiment_label} Sentiment")
                plt.xticks(rotation=45)
                st.pyplot()

            # Display pie chart
            st.write("Pie chart:")
            sentiment_counts = data['sentiment_label'].value_counts()
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot()

            # Display bar chart
            st.write("Bar chart:")
            st.bar_chart(data['sentiment_label'].value_counts())

    elif choice == "Model & Prediksi":
        st.title("Model & Prediksi")

        data = st.file_uploader("Unggah Data yang Telah Diproses", type=["csv"])

        if data is not None:
            data = pd.read_csv(data)

            # Mengatasi nilai NaN atau null pada kolom cleaned_text
            data['cleaned_text'].fillna('', inplace=True)

            # Muat stopwords
            stopwords = load_stopwords()

            # Fitur kata negatif
            negative_words = load_negative_words()
            positive_words = load_positive_words()

            # Feature Extraction
            vectorizer = TfidfVectorizer(max_features=1000)
            features = vectorizer.fit_transform(data['cleaned_text'])

            # Prepare data for modeling
            X = features
            y = data['sentiment_label']

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Latih KNN classifier
            knn_classifier = KNeighborsClassifier(n_neighbors=5)
            knn_classifier.fit(X_train, y_train)

            # Make predictions
            y_pred = knn_classifier.predict(X_test)

            # Display accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Akurasi: {accuracy:.2f}")

            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Display confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=knn_classifier.classes_,
                        yticklabels=knn_classifier.classes_)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show()

            # Display classification report
            classification_rep = classification_report(y_test, y_pred)
            st.text("Classification Report:")
            st.text(classification_rep)

            # Form untuk prediksi
            st.write("Form Prediksi:")
            input_text = st.text_input("Masukkan komentar:")
            if st.button("Prediksi"):
                cleaned_input = clean_text(input_text, stopwords)
                input_features = vectorizer.transform([cleaned_input])

                # Periksa apakah kata negatif ada dalam komentar
                negative_word_found = any(word in cleaned_input.split() for word in negative_words)
                positive_word_found = any(word in cleaned_input.split() for word in positive_words)
                

                if negative_word_found:
                    prediction = "Negative"
                elif positive_word_found:
                    prediction = "Positive"
                else:
                    prediction = knn_classifier.predict(input_features)[0]

                st.write(f"Sentimen yang Diprediksi: {prediction}")


    elif choice == "About":
        st.title("Tentang Aplikasi Sentiment Analisis Instagram")
        st.write("Aplikasi ini digunakan untuk melakukan analisis sentimen pada komentar di platform Instagram.")
        st.write("Dengan aplikasi ini, Anda dapat melakukan berbagai tugas seperti:")
        st.write("- Melihat data komentar berdasarkan topik yang telah dipilih.")
        st.write("- Melakukan preprocessing pada data komentar, termasuk membersihkan teks dan mengidentifikasi sentimen.")
        st.write("- Melihat visualisasi berupa wordcloud, diagram batang, dan pie chart dari data komentar.")
        st.write("- Melakukan prediksi sentimen menggunakan model K-Nearest Neighbors (KNN).")
        st.write("Aplikasi ini dibangun dengan menggunakan Streamlit dan beberapa pustaka analisis teks populer.")
        st.write("Selamat menggunakan aplikasi ini untuk menjalankan analisis sentimen pada data Instagram!")


if __name__ == "__main__":
    main()