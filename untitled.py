import flask
import praw
import datetime

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
        title = submission.title
        author = submission.author
        date = datetime.datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        score = submission.score
        flair = submission.link_flair_css_class
        return flask.render_template('main.html', original_input={'Title':title, 'Author':author, 'Date':date, 'Score':score}, result=flair,)

if __name__ == '__main__':
	# app.debug=True
	app.run()