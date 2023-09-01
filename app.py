import string
import requests
import streamlit as st
import re
import csv
import base64
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')


# Halaman "Home"
st.set_page_config(
    page_title="SENTIMENT ANALISIS TWITTER",
    # Replace with your desired icon URL or emoji
    page_icon="logoig.png",
    layout="wide"
)

# hide menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""


def preprocess_text(text):
    # remove URL
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    # remove hashtags
    text = re.sub(r'#', '', text)

    # remove mention handle user (@)
    text = re.sub(r'@[\w]*', ' ', text)

    # remove punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in text.lower():
        if x in punctuations:
            text = text.replace(x, " ")

    # remove extra whitespace
    text = text.strip()

    # lowercase
    text = text.lower()
    return text


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


# Provide a default value for 'stopwords'
def analyze_sentiment(text, stopwords=None):
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


def load_stopwords():
    path_stopwords = [
        "https://raw.githubusercontent.com/ramaprakoso/analisis-sentimen/master/kamus/stopword.txt",
        "https://raw.githubusercontent.com/yasirutomo/python-sentianalysis-id/master/data/feature_list/stopwordsID.txt",
        "https://raw.githubusercontent.com/onlyphantom/elangdev/master/elang/word2vec/utils/stopwords-list/fpmipa-stopwords.txt",
        "https://raw.githubusercontent.com/onlyphantom/elangdev/master/elang/word2vec/utils/stopwords-list/sastrawi-stopwords.txt",
        "https://raw.githubusercontent.com/onlyphantom/elangdev/master/elang/word2vec/utils/stopwords-list/aliakbars-bilp.txt",
        "https://raw.githubusercontent.com/onlyphantom/elangdev/master/elang/word2vec/utils/stopwords-list/pebbie-pebahasa.txt",
        "https://raw.githubusercontent.com/onlyphantom/elangdev/master/elang/word2vec/utils/stopwords-list/stopwords-id.txt"
    ]

    stopwords_l = stopwords.words('indonesian')
    for path in path_stopwords:
        response = requests.get(path)
        stopwords_l += response.text.split('\n')

    custom_stopwords = '''
    yg yang dgn ane smpai bgt gua gwa si tu ama utk udh btw
    ntar lol ttg emg aj aja tll sy sih kalo nya trsa mnrt nih
    ma dr ajaa tp akan bs bikin kta pas pdahl bnyak guys abis tnx
    bang banget nang mas amat bangettt tjoy hemm haha sllu hrs lanjut
    bgtu sbnrnya trjadi bgtu pdhl sm plg skrg
    '''

    return set(stopwords_l + custom_stopwords.split())


with st.sidebar:
    st.image(Image.open('logoig.png'))
    st.caption('© SoorngDev 2023')

# Konfigurasi Pilihan Menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Dataset", "Processing", "Modeling", "Prediksi"],
    icons=["house", "book", "graph-up", "graph-up", "graph-down"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Menu Sentimen Berita
if selected == "Home":
   # Define custom CSS style to center the text
    st.markdown(
        f"""
        <style>
            .centered-text {{
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 30px;
                margin-bottom:30px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='text-align: center; color: white;'>Aplikasi Analisis Sentimen Instagram</h1>", unsafe_allow_html=True)

    st.image("https://www.travelmediagroup.com/wp-content/uploads/2022/04/bigstock-Market-Sentiment-Fear-And-Gre-451706057-2880x1800.jpg", use_column_width=True)

# Menu Sentimen Pasar
if selected == "Dataset":
    st.title("DataSet")

    def load_data(file_path):
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat data: {e}")
            return None

    # Dropdown untuk memilih topik
    topik_list = ["Prabowo", "Sandiaga Uno", "Airlangga Hartarto"]
    topic = st.selectbox("Pilih topik:", topik_list, index=0)

    # Tampilkan tombol untuk memuat data
    load_button = st.button("Muat Data")

    # Pengecekan apakah tombol ditekan
    if load_button:
        if topic == "Prabowo":
            file_path = "dataset/prabowo.csv"
        elif topic == "Sandiaga Uno":
            file_path = "dataset/sandiuno.csv"
        elif topic == "Airlangga Hartarto":
            file_path = "dataset/airlanggahartarto.csv"
        else:
            st.warning(
                "Anda belum memilih topik. Pilih topik untuk menampilkan data.")
            file_path = None

        # Jika file_path telah ditentukan, muat dan tampilkan data
        if file_path is not None:
            st.write(f"Menampilkan data dari {file_path}")
            data = load_data(file_path)
            if data is not None:
                st.dataframe(data)

                # Create the download link for CSV
                csv_data = data.to_csv(
                    index=False, quoting=csv.QUOTE_NONNUMERIC)
                b64 = base64.b64encode(csv_data.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{topic}_data.csv">Download Data CSV</a>'
                st.markdown(href, unsafe_allow_html=True)


if selected == "Processing":
    st.title("Labeling & Processing")

    stop_words = load_stopwords()

    # Membuat objek PorterStemmer
    ps = PorterStemmer()

    def analyze_sentiment(comment):
        analysis = TextBlob(comment)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    def clean_comment(comment):
        # Menghapus karakter yang tidak diinginkan
        comment = re.sub(
            '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', comment.lower())
        return comment

    def case_folding(comment):
        # Mengubah teks menjadi lowercase
        return comment.lower()

    def tokenizing(comment):
        # Melakukan tokenisasi pada teks
        return word_tokenize(comment)

    def normalization(tokens):
        # Melakukan normalisasi kata-kata
        normalized_tokens = []
        for token in tokens:
            normalized_token = ps.stem(token)
            normalized_tokens.append(normalized_token)
        return normalized_tokens

    def removal_stopwords(tokens):
        # Menghapus stopwords dari teks
        tokens_without_stopwords = [
            token for token in tokens if token not in stop_words]
        return tokens_without_stopwords

    def stemming(tokens):
        # Melakukan stemming pada kata-kata
        stemmed_tokens = []
        for token in tokens:
            stemmed_token = ps.stem(token)
            stemmed_tokens.append(stemmed_token)
        return stemmed_tokens

    uploaded_file = st.file_uploader(
        "Pilih file CSV untuk preprocessing", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Info File:")
        st.write("- Nama File: ", uploaded_file.name)
        st.write("Data Awal:")
        st.write(df.head(1000))

        def preprocessing():
            st.write("Start Pre-processing")
            st.write("| cleaning...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['clean_comment'] = df['comment'].apply(clean_comment)

            st.write("| case folding...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['clean_comment'] = df['clean_comment'].apply(case_folding)

            st.write("| tokenizing...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['tokens'] = df['clean_comment'].apply(tokenizing)

            st.write("| normalization...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['normalized_tokens'] = df['tokens'].apply(normalization)

            st.write("| removal stopwords...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['tokens_without_stopwords'] = df['normalized_tokens'].apply(
                removal_stopwords)

            st.write("| stemming...")
            time.sleep(1)  # Simulasi waktu pemrosesan
            df['stemmed_tokens'] = df['normalized_tokens'].apply(stemming)

            st.write("Finish Pre-processing")

        if st.button("Mulai Pre-processing"):

            preprocessing()
            # Drop specified columns
            columns_to_drop = ["profilePictureUrl", "profileUrl", "likeCount",
                               "replyCount", "commentDate", "timestamp", "commentId", "ownerId", "query"]
            df = df.drop(columns=columns_to_drop)
            df['label'] = df['clean_comment'].apply(analyze_sentiment)

            # Mengurutkan label ke paling akhir
            cols = df.columns.tolist()
            cols.remove('label')
            cols.append('label')
            df = df[cols]

            st.write("Hasil Preprocessing:")
            st.write(df.head(1000))

            temp_file = df.to_csv(index=False)
            b64 = base64.b64encode(temp_file.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_preprocessing.csv">Download Hasil Preprocessing</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("Silakan pilih file CSV untuk melakukan preprocessing.")


if selected == "Modeling":
    st.title("Modeling & Visualisasi")
    # Kode untuk menampilkan konten halaman "Modeling"
    uploaded_file = st.file_uploader(
        "Pilih file CSV untuk preprocessing", type=["csv"])

    # Jika file CSV dipilih
    if uploaded_file is not None:

        # Membaca file CSV menjadi DataFrame
        df = pd.read_csv(uploaded_file)

        # Menampilkan informasi tentang file yang diunggah
        st.write("Info File:")
        st.write("- Nama File: ", uploaded_file.name)

        # # Menampilkan data awal dari DataFrame
        st.write("Data Awal:")
        st.write(df.head(1000))

        # Fungsi untuk melakukan analisis sentimen menggunakan Logistic Regression

        # Menambahkan tombol untuk memulai preprocessing dan training model
        if st.button("Mulai Preprocessing dan Training Model"):

            # Memisahkan fitur dan label
            X = df['clean_comment']
            y = df['label']

            # Melakukan vectorization pada teks
            vectorizer = CountVectorizer()
            X_vectorized = vectorizer.fit_transform(X)

            # Membagi data menjadi data latih dan data uji
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y, test_size=0.2, random_state=42)

            # Latih KNN classifier
            knn_classifier = KNeighborsClassifier(n_neighbors=5)
            knn_classifier.fit(X_train, y_train)

           # Make predictions
            y_pred = knn_classifier.predict(X_test)

            # Menghitung akurasi, F1 score, dan confusion matrix model
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            # Membuat classification report
            classification_rep = classification_report(y_test, y_pred)

            # Menampilkan akurasi, F1 score, dan confusion matrix
            st.write("Accuracy: {:.2f}%".format(accuracy * 100))
            st.write("Precision: {:.2f}%".format(precision * 100))
            st.write("Recall: {:.2f}%".format(recall * 100))
            st.write("F1-score: {:.2f}%".format(f1 * 100))

            # Menampilkan classification report
            st.text("Classification Report:")
            st.text(classification_rep)

            fig, ax = plt.subplots(figsize=(10, 3))
            # Visualisasi confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            plt.tight_layout()

            # Menampilkan visualisasi data
            st.pyplot(fig)

            # Menggabungkan semua teks menjadi satu string
            all_text = ' '.join(df['clean_comment'].values)

            # Membuat objek WordCloud
            wordcloud = WordCloud(width=500, height=100, max_words=150,
                                  background_color='white').generate(all_text)

            # Visualisasi WordCloud
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title('WordCloud - Kata yang Sering Muncul')
            ax.axis('off')

            # Menampilkan visualisasi data
            st.pyplot(fig)

            # Menghitung kata yang sering muncul
            from collections import Counter

            # Mengubah string menjadi list kata-kata
            words_list = all_text.split()

            # Menghitung jumlah kemunculan setiap kata
            word_count = Counter(words_list)

            # Mengambil 10 kata yang paling sering muncul
            common_words = word_count.most_common(10)

            # Mengambil kata dan jumlah kemunculannya
            words = [word for word, count in common_words]
            count = [count for word, count in common_words]

            # Menggabungkan semua teks tweet positif dan negatif menjadi satu string
            all_positive_tweets = ' '.join(
                df[df['label'] == 'Positive']['clean_comment'])
            all_negative_tweets = ' '.join(
                df[df['label'] == 'Negative']['clean_comment'])

            a = len(df[df["label"] == "Positive"])
            b = len(df[df["label"] == "Negative"])
            c = len(df[df["label"] == "Neutral"])
            # d = len(df[df["label"] == "Mixed"])

            # Membuat diagram batang
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(["Positive", "Negative", "Neutral"], [a, b, c])
            ax.set_title('Jumlah Data untuk Setiap Sentimen')
            ax.set_xlabel('Sentimen')
            ax.set_ylabel('Jumlah Data')
            # Menampilkan visualisasi data
            st.pyplot(fig)

            # Membuat diagram batang
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(words, count)
            ax.set_title('Kata yang Sering Muncul')
            ax.set_xlabel('Kata')
            ax.set_ylabel('Jumlah Kemunculan')
            # Menampilkan visualisasi data
            st.pyplot(fig)

        # Piechart
            a = len(df[df["label"] == "Positive"])
            b = len(df[df["label"] == "Negative"])
            c = len(df[df["label"] == "Neutral"])
            labels = ['Positive', 'Negative', 'Neutral']
            sizes = [a, b, c]
            colors = ['#66b3ff', '#ff9999', '#99ff99']
            fig1, ax1 = plt.subplots(figsize=(10, 3))
            ax1.pie(sizes, colors=colors, labels=labels,
                    autopct='%1.1f%%', startangle=90)
            # Draw circle
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax1.axis('equal')
            plt.title("Persentase Sentimen")
            plt.tight_layout()
            # Menampilkan visualisasi data
            st.pyplot(fig1)

            # Membuat WordCloud untuk tweet positif

            wordcloud_positive = WordCloud(
                width=500, height=100, background_color='white').generate(all_positive_tweets)
            plt.figure(figsize=(10, 3))
            plt.imshow(wordcloud_positive, interpolation='bilinear')
            plt.axis('off')
            plt.title('WordCloud untuk Tweet Positif')
            # Menampilkan visualisasi data
            st.pyplot(plt)

            # Mengecek apakah ada sentimen negatif
            if 'Negative' in df['label'].values:

                wordcloud_negative = WordCloud(
                    width=500, height=100, background_color='white').generate(all_negative_tweets)
                plt.figure(figsize=(10, 3))
                plt.imshow(wordcloud_negative, interpolation='bilinear')
                plt.axis('off')
                plt.title('WordCloud untuk Tweet Negatif')
                # Menampilkan visualisasi data
                st.pyplot(plt)

if selected == "Prediksi":
    st.title("Prediksi Sentimen")
    # Kode untuk menampilkan konten halaman "Modeling"
    uploaded_file = st.file_uploader(
        "Upload File Hasil Processing", type=["csv"])

    # Jika file CSV dipilih
    if uploaded_file is not None:

        # Membaca file CSV menjadi DataFrame
        df = pd.read_csv(uploaded_file)

        # Menampilkan informasi tentang file yang diunggah
        st.write("Info File:")
        st.write("- Nama File: ", uploaded_file.name)

        # # Menampilkan data awal dari DataFrame
        st.write("Data Awal:")
        st.write(df.head(1000))
       # Mengatasi nilai NaN atau null pada kolom cleaned_text
        df['clean_comment'].fillna('', inplace=True)

        # Muat stopwords
        stopwords = load_stopwords()

        # Fitur kata negatif
        negative_words = load_negative_words()
        positive_words = load_positive_words()

        # Feature Extraction
        vectorizer = TfidfVectorizer(max_features=1000)
        features = vectorizer.fit_transform(df['clean_comment'])

        # Prepare data for modeling
        X = features
        y = df['label']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # Latih KNN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = knn_classifier.predict(X_test)
        # Form untuk prediksi
        st.write("Form Prediksi:")
        input_text = st.text_input("Masukkan komentar:")
        if st.button("Prediksi"):
            cleaned_input = clean_text(input_text, stopwords)
            input_features = vectorizer.transform([cleaned_input])

            # Periksa apakah kata negatif ada dalam komentar
            negative_word_found = any(
                word in cleaned_input.split() for word in negative_words)
            positive_word_found = any(
                word in cleaned_input.split() for word in positive_words)

            if negative_word_found:
                prediction = "Negative"
            elif positive_word_found:
                prediction = "Positive"
            else:
                prediction = knn_classifier.predict(input_features)[0]

            st.write(f"Sentimen yang Diprediksi: {prediction}")
