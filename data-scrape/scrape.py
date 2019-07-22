import pandas as pd
import requests
import json
import time
import datetime

def getPushShiftData(after, before):
	url = 'https://api.pushshift.io/reddit/search/submission/?subreddit=india&sort=desc&sort_type=created_utc&after=' + str(after) + '&before=' + str(before) + '&size=1000'
	print(url)
	r = requests.get(url)
	data = json.loads(r.text)
	return data['data']

after = 1560556800
before = 1560643199
len_day = before - after

my_data = {
	'data': []
}

feature = ['link_flair_css_class', 'author', 'author_flair_css_class', 'created_utc', 'domain', 'is_crosspostable', 'is_meta', 'is_original_content', 'is_self', 'is_video', 'num_comments', 'num_crossposts', 'full_link', 'media_only', 'no_follow', 'over_18', 'parent_whitelist_status', 'pinned', 'score', 'total_awards_received', 'title', 'send_replies', 'spoiler', 'selftext', 'url']

for i in range(4000):
	print(i)
	temp_data = getPushShiftData(after, before)
	cleaned_data = []
	for post in temp_data:
		dct = {key: post[key] if key in post else None for key in feature}
		if dct['link_flair_css_class'] == None:
			continue
		if dct['created_utc'] != None:
			dct['created_utc'] = datetime.datetime.utcfromtimestamp(dct['created_utc']).strftime('%Y-%m-%d %H:%M:%S')
		cleaned_data.append(dct)		

	my_data['data'].extend(cleaned_data)
	
	if i % 100 == 99:
		with open('cleaned_data2.json', 'w') as json_file:
			json.dump(my_data, json_file)

	after -= len_day
	before -= len_day
