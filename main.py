import flask
import praw
import datetime
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#import  textblob, string
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

# authentication
reddit = praw.Reddit(client_id='wwoXPcv380hcfQ', 
					client_secret='YrJomVhbTWYXFjnU_VGk-K7seho', 
					user_agent='raghavflair')

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))
	
	if flask.request.method == 'POST':        
	
		post_url = flask.request.form['post_url']
		submission = reddit.submission(url = post_url)
		feature = {'link_flair_css_class': [submission.link_flair_css_class], 
					'author': [submission.author], 
					'author_flair_css_class': [submission.author_flair_css_class], 
					'created_utc': [datetime.datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')], 
					'domain': [submission.domain], 
					'is_crosspostable': [submission.is_crosspostable], 
					'is_meta': [submission.is_meta], 
					'is_original_content': [submission.is_original_content], 
					'is_self': [submission.is_self], 
					'is_video': [submission.is_video], 
					'num_comments': [submission.num_comments], 
					'num_crossposts': [submission.num_crossposts], 
					'full_link': [post_url], 
					'media_only': [submission.media_only], 
					'no_follow': [submission.no_follow], 
					'over_18': [submission.over_18], 
					'parent_whitelist_status': [submission.parent_whitelist_status], 
					'pinned': [submission.pinned], 
					'score': [submission.score], 
					'total_awards_received': [submission.total_awards_received], 
					'title': [submission.title], 
					'send_replies': [submission.send_replies], 
					'spoiler': [submission.spoiler], 
					'selftext': [submission.selftext], 
					'url': [submission.url]}

		df = pd.DataFrame(feature)

		#convert unix time to datetime stamp and get time features
		df['created_utc'] =  pd.to_datetime(df['created_utc'], format='%Y-%m-%d %H:%M:%S')
		df['month'] =  df['created_utc'].dt.month
		df['year'] =  df['created_utc'].dt.year
		df['day'] =  df['created_utc'].dt.day
		df['hour'] =  df['created_utc'].dt.hour
		df['weekday'] =  df['created_utc'].dt.weekday
		df['weekday_name'] =  df['created_utc'].dt.day_name()

		# drop features which doesnt have variability
		df.drop(['author_flair_css_class', 'is_meta', 'is_original_content','link_flair_css_class', \
				 'media_only', 'no_follow', 'pinned', 'total_awards_received', 'url'], \
				 axis=1, inplace = True)


		#encode flair
		flair_label_enc = pickle.load(open('flair_label_enc.sav', 'rb'))

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


		wnl = WordNetLemmatizer()
		Stop_Words = set(stopwords.words('english'))
		Question_Words = ['what','which','who','whom','when''where','why','how']
		Stop_Words_No_Question = [ w for w in Stop_Words if w not in Question_Words]
		EMBEDDING_DIM = 300

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


		#MAX_SEQUENCE_LENGTH = max([len(x.split(" ")) for x in df['title_selftext'] ])
		MAX_SEQUENCE_LENGTH = 200

		df['title_selftext'] = df['title_selftext'].apply(preprocess)
		freq_words_list = pickle.load(open('freq_words_list.pkl', 'rb'))


		df['title_selftext'] = df['title_selftext'].apply(lambda x: " ".join(x for x in x.split() \
								if x in freq_words_list))
		tokenizer = pickle.load(open('tokenizer.sav', 'rb'))

		data_1 = pad_sequences(tokenizer.texts_to_sequences(df['title_selftext']), maxlen=200)

		nb_words = len(word_index) + 1
		print('Vocabulary '+str(nb_words))

		df['full_link_text'] = df['full_link_text'].apply(lambda x:  ' '.join([w for w in x.split('_')]))
		df['full_link_text'] = df['full_link_text'].apply(lambda x: " ".join(x for x in x.split() \
								if x in freq_words_list))


		data_2 = pad_sequences(tokenizer.texts_to_sequences(df['full_link_text']), maxlen=20)

		num_comments_scale = pickle.load(open('num_comments_scale.sav', 'rb'))

		df['num_comments_scale'] = num_comments_scale.transform(df[['num_comments']]) 
		  
		score_scale = StandardScaler()
		score_scale = pickle.load(open('score_scale.sav', 'rb'))
		df['score_scale'] = score_scale.transform(df[['score']])   


		lbl_domain = pickle.load(open('lbl_domain.sav', 'rb'))

		df['domain_enc'] = lbl_domain.transform(df['domain'])

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


		model.load_model('modelcheckpoint.h5')

		preds = model.predict([X[:,:200], X[:,200:220], X[:,220:]],\
							 batch_size=128, verbose=1)

		#model.save('./'+key+'_'+'model.h5')

		preds_label_enc = np.argmax(preds, axis=1)

		preds_label = list(flair_label_enc.inverse_transform(preds_label_enc))

		return flask.render_template('main.html', original_input = feature, result = preds_label,)

if __name__ == '__main__':
	# app.debug=True
	app.run()