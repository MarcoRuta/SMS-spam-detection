import nltk
import string

nltk.download('stopwords')
nltk.download('punkt')

punctuations = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

def stem(text):
    if not text:
        return []

    # Tokenize the message
    tokens = nltk.word_tokenize(text)

    # Remove punctuation from tokens
    tokens = [i.strip("".join(punctuations)) for i in tokens if i not in punctuations]

    # Remove stopwords and stem tokens
    if len(tokens) > 2:
        return [stemmer.stem(w) for w in tokens if w not in stopwords]
    return []