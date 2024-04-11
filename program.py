import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk.data

# Function for text preprocessing
def preprocess(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    return text.strip()

# Function to load the dataset
def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)

# Function to get article text by ID
def get_article_text(article_id, dataset):
    return dataset.loc[article_id]['Text']

# Function to split text into sentences
def split_into_sentences(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(text)

class ArticleIndex:
    def __init__(self):
        # Initialize an index to store the text of each article and a TF-IDF vectorizer
        self.index = defaultdict(list)  # Index to store text of each article
        self.vectorizer = TfidfVectorizer(preprocessor=preprocess)  # TF-IDF vectorizer for text processing

    def index_article(self, article_id, text):
        # Add the text of the current article to the index and fit the TF-IDF vectorizer to it
        self.index[article_id] = text
        self.vectorizer.fit_transform([text])  # Fit the TF-IDF vectorizer to the text of the current article

    def search(self, query):
        # Transform the query text into a TF-IDF vector using the pre-fitted vectorizer
        query_vec = self.vectorizer.transform([query])
        similarities = {}
        # Calculate cosine similarity between the query vector and each article vector in the index
        for article_id, text in self.index.items():
            article_vec = self.vectorizer.transform([text])
            similarity = cosine_similarity(query_vec, article_vec)[0][0]
            similarities[article_id] = similarity
        # Sort the similarities in descending order and return the sorted list of (article_id, similarity) tuples
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_similarities

class ArticleRetrievalSystem:
    def __init__(self, article_index, dataset):
        self.article_index = article_index
        self.dataset = dataset

    def retrieve_fragments(self, query):
        relevant_articles = self.article_index.search(query)  # Retrieve relevant articles based on the query
        fragments = []
        for article_id, similarity in relevant_articles:
            article_text = get_article_text(article_id, self.dataset)  # Get the text of the current article
            sentences = split_into_sentences(article_text)  # Split the article text into sentences
            query_word = re.compile(r'\b{}\b'.format(re.escape(query)), flags=re.IGNORECASE)  # Compile regex pattern for the query word
            for i, sentence in enumerate(sentences):
                if query_word.search(sentence):  # Check if the query word appears in the current sentence
                    start_index = max(0, i - 1)  # Index of the sentence before the one containing the query word
                    end_index = min(len(sentences), i + 2)  # Index of the sentence after the one containing the query word
                    fragment = ' '.join(sentences[start_index:end_index])  # Extract the fragment containing the query word
                    article_title = self.dataset.loc[article_id]['Title']  # Get the title of the current article
                    fragments.append((fragment, article_title))  # Add the fragment and article title to the list of fragments
        return fragments

# Sample usage
if __name__ == "__main__":
    dataset_path = "medium.csv"
    dataset = load_dataset(dataset_path)

    article_index = ArticleIndex()
    for article_id, row in dataset.iterrows():
        article_index.index_article(article_id, row['Text'])  # Index each article in the dataset

    retrieval_system = ArticleRetrievalSystem(article_index, dataset)
    query = input("Wprowadź słowo kluczowe: ")  # Getting the keyword from user
    fragments = retrieval_system.retrieve_fragments(query)

    # Generate HTML file
    with open("fragments.html", "w") as f:
        f.write("<html><head><title>Fragmenty artykułów</title></head><body>")
        for fragment, article_title in fragments:
            # Highlight the keyword in each fragment and add the article title at the end in bold
            highlighted_fragment = re.sub(r'\b' + query + r'\b', r'<span style="background-color: yellow;">\g<0></span>', fragment, flags=re.IGNORECASE)
            f.write(f"<p>{highlighted_fragment} <strong>{article_title}</strong></p>")  # Write the fragment and article title to HTML file
        f.write("</body></html>")  # Close HTML tags