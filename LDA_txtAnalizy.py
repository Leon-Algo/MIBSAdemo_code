import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import jieba


# Load the text file
file_path = "E:/13届市调分析大赛(研究生)/nb的评语.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.readlines()

# Preprocessing: tokenize and remove stopwords
docs = []
stopwords = set(['的', '了', '在', '；',  '也', '有', '都', '能', '会', '不', '...', '我们', '可以', ' ', '“', '？', '-', '—', ',', '”', '和', '。', '，', '、', '：', '（', '）', '.', '/', '是', '它', '我', ':', '就', '你', '他'])  # add your own list of stopwords
for line in content:
    words = [word for word in jieba.cut(line.strip()) if word not in stopwords]
    docs.append(words)

# Create a dictionary from the tokenized documents
dictionary = corpora.Dictionary(docs)

# Convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Train the LDA model
num_topics = 10  # adjust this based on the number of topics you want to extract
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=num_topics,
                     passes=15,
                     alpha='auto',
                     per_word_topics=True)

# Print the top words for each topic
for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
    print(f"Topic {i}: {topic}")