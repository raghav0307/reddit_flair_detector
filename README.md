# Reddit Flair Detector

A Reddit Flair Detector system to detect flairs (category) of India subreddit submissions (posts). The application has been deployed online using Heroku at [Reddit Flair Detector](https://reddit-flair-predictor.herokuapp.com).

### Directory Structure

The directory is a **Flask** web application set-up for hosting on *Heroku* servers. The description of files and folders can be found below:
.  
├── [data-scrape](https://github.com/raghav0307/reddit_flair_detector/tree/master/data-scrape)  
│   └── [scrape.py](https://github.com/raghav0307/reddit_flair_detector/blob/master/data-scrape/scrape.py)  
├── [main.py](https://github.com/raghav0307/reddit_flair_detector/tree/master/main.py)  
├── [Procfile](https://github.com/raghav0307/reddit_flair_detector/tree/master/Procfile)  
├── [README.md](https://github.com/raghav0307/reddit_flair_detector/tree/master/README.md)  
├── [requirements.txt](https://github.com/raghav0307/reddit_flair_detector/tree/master/requirements.txt)  
└── [templates](https://github.com/raghav0307/reddit_flair_detector/tree/master/templates)  
&nbsp; &nbsp; &nbsp;└── [main.html](https://github.com/raghav0307/reddit_flair_detector/blob/master/templates/main.html)  


## Deep Neural Network with Seq to Seq multi head input architecture for Flair Detection

I started off with text cleaning. I found that the target feature ‘link_flair_css_class’ contains lots of junk. I
cleaned it. Label encode the target class. Merged the ‘Self text’ with ‘title’. Convert the the time to
timestamp. Created following Sklearn model
* Tree Based Model
	* DecisionTreeClassifier
	* ExtraTreeClassifier
* Neighbors Model
	* KNeighborsClassifier
* Ensemble Model
	* GradientBoostingClassifier
	* RandomForestClassifier
	* ExtraTreesClassifier

and used f1 score as metrics and build above model on following features:
* Bag of words count
* Tfidf count
* tfidf_vect_ngram_chars ngram(2,3)

got similar score of around 55 f1 score. I also created post tag features:
* noun count
* verb_count
* Adj count
* Pron count
* Word density
* Punctuation count

Then I realised that I’m not exploiting the text semantic in above model. I decide to use Deep Neural
Network with Seq to Seq learning with multi head architecture. The best thing about this is that now I
don’t need to extensive features engineering. Deep Neural Network is strong enough to learn features in
its own.
I used three input head to feed input as detailed below:
* Title and Self text merge together
* Full Link
* Non text features
	* Label encoded Domain
	* StandardScale Score
	* StandardScale Number of Comments
	* is_video
	* is_crosspostable
	* is_self
	* over_18
	* parent_whitelist_status
	* send_replies
	* Timestamp extracted features i.e. month, day, weekday, hours

Other features which we got it from Reddit API were dropped as they don’t have much variability .
I cleaned and normalised the test before feeding into embedding layer.
I used Stanford Glove embedding. Initially I used ‘glove.840B.300d.txt&#39; but I was not able to save model
‘coz of size of non trained parameter. Then I decide to use ‘glove.twitter.27B.100d.txt’ . I choose this
because it close to Reddit post textual data.

I used most frequent word to decide the size of the Vocabulary for word embedding. This took lot of time
to adjust to teh size which mu laptop can handle data.
I limit the text sequence size to 200.
I used GRU model as it has less parameter than LSTM so it will train quickly with limited time. I used 5
epoch to train the model and saved the best model.

