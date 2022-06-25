import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk,re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

df = pd.read_csv("ACS_training_data.csv")

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
stop_words = stopwords.words('english')

def clean_text(text):
  text = text.lower()
  text = REPLACE_BY_SPACE_RE.sub('', text)
  text = BAD_SYMBOLS_RE.sub('', text) 
  return text

# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None

def lemmatize_sentence(sentence):
  #tokenize the sentence and find the POS tag for each token
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
  #tuple of (token, wordnet_tag)
  wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
  lemmatized_sentence = []
  for word, tag in wordnet_tagged:
    if tag is None:
      #if there is no available tag, append the token as is
      lemmatized_sentence.append(word)
    else:        
      #else use the tag to lemmatize the token
      lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
  return " ".join(lemmatized_sentence)

def get_cosine(query,corpus):
  query = lemmatize_sentence(query)
  print(query)
  query = clean_text(query)
  corpus = lemmatize_sentence(corpus)
  corpus = clean_text(corpus)
  query_word = word_tokenize(query)
  corpus_word = word_tokenize(corpus)
  query_ = {word for word in query_word if word not in stop_words}
  corpus_ = {word for word in corpus_word if word not in stop_words}
  query_valid_list = []
  corpus_valid_list = []
  
  sentence_vector = query_.union(corpus_)
  
  for word in sentence_vector: 
    if word in query_:
      query_valid_list.append(1) 
    else:
      query_valid_list.append(0) 
    if word in corpus_:
      corpus_valid_list.append(1) 
    else:
      corpus_valid_list.append(0)

  product_numerator = 0
  for i in range(len(sentence_vector)):
    product_numerator+= query_valid_list[i]*corpus_valid_list[i]
  try:
    cosine_result = float(product_numerator) / float((sum(query_valid_list)*sum(corpus_valid_list))**0.5)
    return cosine_result
  except:
    return 0

def get_response(q_point):
  score_list = [get_cosine(q_point,j) for j in (df["QUESTIONS"])]
  return df["ANSWERS"][score_list.index(max(score_list))]
