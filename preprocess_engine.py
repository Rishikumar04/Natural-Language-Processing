
import nltk
import spacy
import re
from bs4 import BeautifulSoup
import requests
import unicodedata
import contractions

nlp = spacy.load('en_core_web_sm')

def remove_special_characters(text, remove_digits= False):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"pic\S+", "", text)
    pattern = r'[^\w]+' if not remove_digits else r'[^a-zA-Z]'
    text = re.sub(pattern," ",text)
    return text
    
def strip_html_tags(text, remove_digits = False):
    soup = BeautifulSoup(text,'html.parser')
    [s.extract() for s in soup(['iframe','script'])]
    stripped_text = soup.get_text()
    stripped_text = remove_special_characters(stripped_text,remove_digits=remove_digits)
    #stripped_text = re.sub(pattern,' ',stripped_text)
    stripped_text = re.sub(r'\s+',' ', stripped_text)
    return stripped_text

def remove_symbols(text, remove_digits=True):
    text = remove_special_characters(text=text,remove_digits=remove_digits)
    #stripped_text = re.sub(pattern,' ',text)
    stripped_text = re.sub(r'\s+',' ', text)
    return stripped_text

def remove_accented_characters(text):
    text =  unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode('utf-8','ignore')
    return text

def spacy_lemma(text): 
    text = nlp(text)
    new_text = []
    words = [word.lemma_ for word in text]
    for small in words:
        if small == '-PRON-':
            pass
        else:
            new_text.append(small)
    return ' '.join(new_text)

def nltk_stemming(text, stemmer):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def contractions_text(text):
    return contractions.fix(text)

def lemmatizer_nltk(text):
    nltk_lemma = nltk.stem.WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged_pos = nltk.pos_tag(tokens)
    tag_map = {'j': nltk.corpus.wordnet.ADJ,'v': nltk.corpus.wordnet.VERB,'n': nltk.corpus.wordnet.NOUN,'r':              nltk.corpus.wordnet.ADV}
    tagged_tokens = [(word,tag_map.get(tag[0].lower(),nltk.corpus.wordnet.NOUN)) for word, tag in tagged_pos]
    lemmatized_text = " ".join([nltk_lemma.lemmatize(word,tag) for word,tag in tagged_tokens])
    return lemmatized_text

def stop_words_removal(text, is_lower_case = True, stopwords = None):
    if stopwords == None:
        stopwords = nlp.Defaults.stop_words
    
    if not is_lower_case:
        text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if len(word)>1] 
    
    removed_text = [word for word in tokens if word not in stopwords]
    
    return ' '.join(removed_text)

def preprocessor_engine(text, html_strip,  accent_characters, fix_contract, remove_stop_words, stop_words=None, stemmer = nltk.porter.PorterStemmer(), lower=True, to_spacy_lemma=True, to_stem=False, to_lemma=False, remove_digits=False):
    
    if accent_characters:
        text = remove_accented_characters(text)
        
    if fix_contract:
        text = contractions_text(text)
        
    if html_strip:
        text = strip_html_tags(text, remove_digits=remove_digits)
    else:
        text = remove_symbols(text, remove_digits=remove_digits) 
    
    
        
    #if special_char_removal:
        #text = remove_special_characters_2(text, remove_digits)
        
  
        
    if to_spacy_lemma:
        text = spacy_lemma(text)
        
    elif to_stem:
        text = nltk_stemming(text,stemmer)
    
    elif to_lemma:
        text = lemmatizer_nltk(text)
        
    if remove_stop_words:
        if stop_words:
            text = stop_words_removal(text, is_lower_case = lower)
        
        else:
            text =  stop_words_removal(text, is_lower_case = lower,stopwords = stop_words)
    
    return text
