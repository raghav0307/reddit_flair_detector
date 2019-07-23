# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:59:56 2019


"""
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#import  textblob, string
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import os
#os.chdir('F:\\Tech\\DS\\python\\raghav\\')

# read data and create dataframe
data = []
with open('../data/cleaned_data.json', 'r') as json_file:
	data = json.load(json_file)
df = pd.DataFrame(data['data'])

#convert unix time to datetime stamp and get time features
df['created_utc'] =  pd.to_datetime(df['created_utc'], format='%Y-%m-%d %H:%M:%S')
df['month'] =  df['created_utc'].dt.month
df['year'] =  df['created_utc'].dt.year
df['day'] =  df['created_utc'].dt.day
df['hour'] =  df['created_utc'].dt.hour
df['weekday'] =  df['created_utc'].dt.weekday
df['weekday_name'] =  df['created_utc'].dt.day_name()

# drop features which doesnt have variability
df.drop(['author_flair_css_class', 'is_meta', 'is_original_content',\
         'media_only', 'no_follow', 'pinned', 'total_awards_received', 'url'], \
         axis=1, inplace = True)

#clean Flair text
di = {'Entertainment Non-Political': 'Entertainment', 'Removed Removed': 'Removed',\
      'six six' : 'six', 'Net-Neutrality': 'netneutrality', 'Removed Removed Removed': 'Removed',\
      'six six six':'six', 'Removed Removed Removed Removed':'Removed', 'serious':'other',\
      'Dominant': 'Dominant Policy', 'Policy gov' : 'Policy', \
      'Removed Removed Removed Removed Removed Removed': 'Removed',\
      'Non-Political Moderated': 'Non-Political', 'AskIndia ask':'AskIndia',\
      'Non-Political two': 'Non-Political', 'Removed Removed Removed Removed Removed': 'Removed',\
      "":'Removed', 'weekly':'Scheduled', 'tech': 'Science Technology', \
      'reddiquette':'Reddiquette', 'Technology':'Science Technology', 'food':'Food',\
      'ask': 'AskIndia'}
df['link_flair_css_class'].replace(di, inplace=True)

# filter data based on current valid flair
valid_flair = ['Politics',
               'Non-Political',
               'Reddiquette',
               'AskIndia',
               'Policy Economy',
               'Business Finance Policy',
               'Science Technology',
               'Scheduled',
               'Sports',
               'Food',
               'Photography']
              
df = df.loc[df['link_flair_css_class'].isin(valid_flair),:]               
df.reset_index(drop=True, inplace=True)


#encode flair
lbl = LabelEncoder()
df['link_flair_css_class_enc'] = lbl.fit_transform(df['link_flair_css_class'])
pickle.dump(lbl, open('../output/flair_label_enc.sav', 'wb'))
#df.to_csv("flair_data.csv", index=False)

#get link text from full_link_text
df['full_link_text'] = df['full_link'].apply(lambda x: x.split('/')[-2])

#clean selftext features
df['selftext'].fillna('removed', inplace = True)
di = {"": 'removed', '[deleted]':'removed', '[removed]':'removed',\
      '\[removed\]':'removed', '.':'removed', '\n':'removed','#':'removed',\
        'Title':'removed', 'Title.':'removed', " ":'removed', '?':'removed',\
        '^':'removed', 'Title says it all.':'removed'       }
df['selftext'].replace(di, inplace=True)

# merge selftext with title
df['title_selftext'] = df[['selftext','title']].apply(lambda x:  \
                        x[1] + x[0] if x[0] != 'removed' else x[1], axis=1)

'''
# create count features
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words = 'english',\
                             max_features = 5000)
df_count = count_vect.fit_transform(df['title_selftext'])

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words = 'english',\
                             max_features=5000)
df_tdidf = tfidf_vect.fit_transform(df['title_selftext'])

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', \
                            ngram_range=(2,3), max_features=5000)

df_tdidf_ngram = tfidf_vect_ngram_chars.fit_transform(df['title_selftext'])


#create nlp features
pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        pos_tag = textblob.TextBlob(x)
        for tup in pos_tag.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

df['noun_count'] = df['title_selftext'].apply(lambda x: check_pos_tag(x, 'noun'))
df['verb_count'] = df['title_selftext'].apply(lambda x: check_pos_tag(x, 'verb'))
df['adj_count'] = df['title_selftext'].apply(lambda x: check_pos_tag(x, 'adj'))
df['adv_count'] = df['title_selftext'].apply(lambda x: check_pos_tag(x, 'adv'))
df['pron_count'] = df['title_selftext'].apply(lambda x: check_pos_tag(x, 'pron'))
df['char_count'] = df['title_selftext'].apply(len)
df['word_count'] = df['title_selftext'].apply(lambda x: len(x.split()))
df['word_density'] = df['char_count'] / (df['word_count']+1)
df['punctuation_count'] = df['title_selftext'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 

# split data into train test 
X_train, X_test, y_train, y_test = train_test_split(df_count, df['link_flair_css_class_enc'], test_size=0.20, \
                            stratify = df['link_flair_css_class_enc'], shuffle=True,random_state=42)

#run algorithm
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report,roc_auc_score

model = DecisionTreeClassifier()

model.fit(X_train, y_train)
predictions_test = model.predict(X_test)

class_names = [i for i in range(df['link_flair_css_class'].nunique())]
actual_label = list(lbl.inverse_transform(class_names))
               
print("Classification report for classifier \n%s\n" % \
      ( classification_report(y_test, predictions_test, target_names= actual_label)))


'''
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,GRU
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.noise import GaussianNoise
from keras.initializers import Constant


wnl = WordNetLemmatizer()
Stop_Words = set(stopwords.words('english'))
Question_Words = ['what','which','who','whom','when''where','why','how']
Stop_Words_No_Question = [ w for w in Stop_Words if w not in Question_Words]
EMBEDDING_DIM = 100

def cutter(word):
    if (len(word) < 2):
        return " "
    elif  (len(word) <4):
        return word
    return wnl.lemmatize(wnl.lemmatize(word, "n"), "v")


def preprocess(string):
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)
    string = re.sub(r"\W+", ' ',string)
    string = ' '.join([cutter(w) for w in string.split()])
    string = ' '.join([w for w in string.split() if w not in Stop_Words_No_Question ])
    return string


def get_embedding():
    embeddings_index = {}
#    GLOVEFILE = 'F:\\embeddings\\glove_word_embedding\\glove.840B.300d.txt'
    GLOVEFILE = '../data/glove.twitter.27B.100d.txt'
    f = open(GLOVEFILE, 'r', encoding='utf-8')
    
    for line in tqdm(f):
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

#MAX_SEQUENCE_LENGTH = max([len(x.split(" ")) for x in df['title_selftext'] ])
MAX_SEQUENCE_LENGTH = 200

df['title_selftext'] = df['title_selftext'].apply(preprocess)
freq = pd.DataFrame(pd.Series(' '.join(df['title_selftext'] ).split()).value_counts(),\
                    columns = ['freq_count'])

freq_words = freq.loc[~freq['freq_count'].isin(range(50))]
freq_words_list = list(freq_words.index)
pickle.dump(freq_words_list, open('../output/freq_words_list.pkl', 'wb'))

df['title_selftext'] = df['title_selftext'].apply(lambda x: " ".join(x for x in x.split() \
                        if x in freq_words_list))
embeddings_index = get_embedding()

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(df['title_selftext'])
pickle.dump(tokenizer, open('../output/tokenizer.sav', 'wb'))

word_index = tokenizer.word_index

data_1 = pad_sequences(tokenizer.texts_to_sequences(df['title_selftext']), maxlen=200)

nb_words = len(word_index) + 1
print('Vocabulary '+str(nb_words))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


df['full_link_text'] = df['full_link_text'].apply(lambda x:  ' '.join([w for w in x.split('_')]))
df['full_link_text'] = df['full_link_text'].apply(lambda x: " ".join(x for x in x.split() \
                        if x in freq_words_list))


data_2 = pad_sequences(tokenizer.texts_to_sequences(df['full_link_text']), maxlen=20)

num_comments_scale = StandardScaler()
df['num_comments_scale'] = num_comments_scale.fit_transform(df[['num_comments']]) 
pickle.dump(num_comments_scale, open('../output/num_comments_scale.sav', 'wb'))
  
score_scale = StandardScaler()
df['score_scale'] = score_scale.fit_transform(df[['score']])   
pickle.dump(score_scale, open('../output/score_scale.sav', 'wb'))


lbl_domain = LabelEncoder()
df['domain_enc'] = lbl_domain.fit_transform(df['domain'])
pickle.dump(lbl_domain, open('../output/domain_enc.sav', 'wb'))

di = {True:1,False:0}

df['is_video']= df['is_video'].map(di).fillna(-1)
df['is_crosspostable']= df['is_crosspostable'].map(di).fillna(-1)
df['is_self']= df['is_self'].map(di).fillna(-1)
df['over_18']= df['over_18'].map(di).fillna(-1)
df['parent_whitelist_status']= df['parent_whitelist_status'].map(di).fillna(-1)
df['send_replies']= df['send_replies'].map(di).fillna(-1)

cols_to_use = ['domain_enc', 'score_scale','num_comments_scale','is_video',\
               'is_crosspostable','is_self', 'over_18','month','day',\
               'parent_whitelist_status','send_replies','weekday','hour']

X = np.concatenate((data_1,data_2, df[cols_to_use].values), axis=1) 
y = df['link_flair_css_class_enc'].values

del(df)
del(embeddings_index)

features_dummy, x_test, label_dummy, y_test = train_test_split(X, y,
                                                    stratify=y, random_state=2019,
                                                    test_size=0.2)
    
x_train, x_valid, y_train, y_valid = train_test_split(features_dummy, label_dummy,
                                                stratify=label_dummy, random_state=2019,
                                                test_size=0.2)

ohc = OneHotEncoder()
y_train_ohc = ohc.fit_transform(y_train.reshape(-1, 1))
y_train_ohc = y_train_ohc.toarray()
y_valid_ohc = ohc.fit_transform(y_valid.reshape(-1, 1))
y_valid_ohc = y_valid_ohc.toarray()
y_test_ohc = ohc.fit_transform(y_test.reshape(-1, 1))
y_test_ohc = y_test_ohc.toarray()


embedding_layer_data1 = Embedding(nb_words,
                        EMBEDDING_DIM,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=200,
                        trainable=False)

embedding_layer_data2 = Embedding(nb_words,
                        EMBEDDING_DIM,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=20,
                        trainable=False)

lstm_layer = GRU(75, recurrent_dropout=0.2)

sequence_1_input = Input(shape=(200,), dtype="int32")
embedded_sequences_1 = embedding_layer_data1(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(20,), dtype="int32")
embedded_sequences_2 = embedding_layer_data2(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)

features_input = Input(shape=(len(cols_to_use),), dtype="float32")
features_dense = Dense(75, activation="relu")(features_input)
features_dense = Dropout(0.2)(features_dense)

merged = concatenate([x1, y1,features_dense])
merged = BatchNormalization()(merged)
merged = GaussianNoise(0.1)(merged)

merged = Dense(150, activation="relu")(merged)
merged = Dropout(0.4)(merged)
merged = BatchNormalization()(merged)

preds = Dense(len(valid_flair), activation='softmax')(merged)

model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], outputs=preds)
model.compile(loss='categorical_crossentropy', \
              optimizer='adam', \
              metrics=['acc'])
key = 'dnn1'
checkpoint = ModelCheckpoint('../output/'+key+'_'+'modelcheckpoint.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')
        #checkpoint = ModelCheckpoint('new1_weights.h5', monitor='val_loss', verbose=1, mode='min')
model.fit([x_train[:,:200], x_train[:,200:220], x_train[:,220:]], \
          y=y_train_ohc, batch_size=16, epochs=5, \
          validation_data=([x_valid[:,:200], x_valid[:,200:220],x_valid[:,220:] ], y_valid_ohc),\
          callbacks=[checkpoint,EarlyStopping(monitor='val_loss', patience=1)])
model.save('../output/'+key+'_'+'model.h5')
#model.load_model('../output/'+key+'_'+'modelcheckpoint.h5')

preds = model.predict([x_test[:,:200], x_test[:,200:220], x_test[:,220:]],\
                     batch_size=128, verbose=1)

#model.save('./'+key+'_'+'model.h5')

preds_label_enc = np.argmax(preds, axis=1)

preds_label = list(lbl.inverse_transform(preds_label_enc))

print("Classification report for classifier \n%s\n" % \
      ( classification_report(y_test, preds_label_enc, target_names= preds_label)))


