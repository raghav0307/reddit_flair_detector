# import pandas as pd
# import requests
# import json

# def getPostDetails(link):
# 	if link[-1] == '/':
# 		link = link[:-1]
# 	url = link + ".json"
# 	print(url)
# 	r = requests.get(url)
# 	print(r.text[:100])
# 	data = r.json
# 	# print(type(data))
# 	# for i in data:
# 	# 	print(i, type(i))
# 	return data

post_url = "https://www.reddit.com/r/india/comments/ce79ay/assam_floods_let_us_stand_by_our_state_in_these/"

# df = pd.DataFrame(getPostDetails(post_url))
# # print(df.to_string())


import praw

# authentication
reddit = praw.Reddit(client_id='wwoXPcv380hcfQ', 
					client_secret='YrJomVhbTWYXFjnU_VGk-K7seho', 
					user_agent='raghavflair')

submission = reddit.submission(url = post_url)
print(submission.title)
print(submission.author)