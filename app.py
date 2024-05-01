from flask import Flask, render_template, request
import nltk
import re
import string
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import operator

app = Flask(__name__)

# Fungsi untuk melakukan peringkasan teks
def summarize_text(title, text):
    # Input teks
    combined_text = title + ". " + text

    def load_stopWords():
        with open('stopword.txt', 'r') as f:
            return [word.strip() for word in f.readlines()]

    listStopword = load_stopWords()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Text Preprocessing judul dan isi
    sentences = sent_tokenize(combined_text)
    output_case_folding = []
    output_tokenizing = []
    output_filtering = []
    output_stemming = []
    documents = []

    for sentence in sentences:
        # Case folding
        sentence_lower = sentence.lower()
        remove_special_chars = re.sub(f"[{string.punctuation}\d]+", " ", sentence_lower)
        output_case_folding.append(remove_special_chars)
        # Tokenizing
        tokens = word_tokenize(remove_special_chars)
        output_tokenizing.append(tokens)
        # Filtering dengan stop words
        filtered_tokens = [token for token in tokens if token not in listStopword]
        output_filtering.append(filtered_tokens)
        # Stemming
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        output_stemming.append(stemmed_tokens)
        documents.append(' '.join(stemmed_tokens))

    preprocessing_data = {
        'Segmentasi kalimat': sentences,
        'Case Folding': output_case_folding,
        'Tokenizing': output_tokenizing,
        'Filtering': output_filtering,
        'Stemming': output_stemming,
    }

    # preprocessing_df = pd.DataFrame(preprocessing_data)
    # preprocessing_df.to_csv('preprocessing_result.csv', index=False)
    # print("Hasil Preprocessing disimpan sebagai 'preprocessing_result.csv'")

    # TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_text = tfidf.fit_transform(documents)
    feature_names = tfidf.get_feature_names_out()

    # tfidf_df = pd.DataFrame(data=tfidf_text.toarray(), columns=feature_names)
    # tfidf_df.to_csv('tfidf_result.csv', index=False)
    # print("\nHasil TF-IDF disimpan sebagai 'tfidf_result.csv'")

    # Cosine similarity title dan teks
    similarity_matrix = cosine_similarity(tfidf_text)
    # similarity_df = pd.DataFrame(similarity_matrix, columns=["Q"] + ["D{}".format(i) for i in range(1, len(sentences))])

    # similarity_df.index = ["Q"] + ["D{}".format(i) for i in range(1, len(sentences))]
    # similarity_df.index.name = "Similarity"

    # print("\nHasil Cosine Similarity:")
    # print(similarity_df)

    # Calculate MMR
    n = 3
    alpha = 0.7
    summary_set = []
    mmr_iterations = []

    similarity_to_title = similarity_matrix[0, 1:]

    while n > 0:
        mmr = {}

        for i, sentence in enumerate(sentences[1:]):
            if sentence not in summary_set:
                sentence_vector = tfidf_text[i + 1]
                similarity_to_summary = [cosine_similarity(sentence_vector, tfidf_text[sentences.index(s)]) for s in summary_set]
                mmr[sentence] = alpha * similarity_to_title[i] - (1 - alpha) * max(similarity_to_summary, default=0)

        # print(f"\nIterasi {len(summary_set) + 1} - Nilai MMR untuk setiap kalimat:")
        # for sentence, mmr_score in mmr.items():
        #     print(f"{sentence}: {mmr_score}")

        selected_sentence = max(mmr.items(), key=operator.itemgetter(1))[0]
        summary_set.append(selected_sentence)
        n -= 1

        mmr_iterations.append({
            'Iteration': len(summary_set),
            'Selected_Sentence': selected_sentence,
            'MMR_Score': mmr[selected_sentence],
            'Summary': ' '.join(summary_set)
        })

    print("\nHasil Ringkasan:")
    for i, sentence in enumerate(summary_set):
        print(f"{i + 1}. {sentence}")

    # mmr_iterations_df = pd.DataFrame(mmr_iterations)
    # mmr_iterations_df.to_csv('mmr_iterations_result.csv', index=False)
    # print("\nHasil MMR per Iterasi disimpan sebagai 'mmr_iterations_result.csv'")

    return ' '.join(summary_set)

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html', summary_result="")

# Route untuk menangani form peringkasan teks
@app.route('/', methods=['POST'])
def summarize():
    title = request.form['title']
    text = request.form['text']

    summary_result = summarize_text(title, text)

    return summary_result

if __name__ == '__main__':
    app.run(debug=True)
