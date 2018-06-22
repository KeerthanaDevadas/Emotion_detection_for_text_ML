from twython import Twython
import io
import sys
def get_tweets(key):
    TWITTER_APP_KEY = 'czA5qEkxIQHRNm4MwLiIAYnsu'  # supply the appropriate value
    TWITTER_APP_KEY_SECRET = 'U76KqNGrnNxTYy3VdMowLcwizpwVwHR8yHQQFm841K30E8jHpp'
    TWITTER_ACCESS_TOKEN = '966360673457508353-1zcN2p22J3hJbm8sI6VSSMm81UVLkVn'
    TWITTER_ACCESS_TOKEN_SECRET = 'to6lFIFrpy3wwPbLTsRlYoGLv7pTi7BgkkYi1OZAV8wOF'

    t = Twython(app_key=TWITTER_APP_KEY,
                app_secret=TWITTER_APP_KEY_SECRET,
                oauth_token=TWITTER_ACCESS_TOKEN,
                oauth_token_secret=TWITTER_ACCESS_TOKEN_SECRET)

    search = t.search(q='#'+key, lang='en')  # **supply whatever query you want here**

    tweets = search['statuses']

    tweets = [i['text'] for i in tweets]
    return tweets
get_tweets('pnb')














#
# for tweet in tweets:
# 	#txt = unicode(tweet['text'],'utf-8')
# 	#print tweet['id_str']+'\n'+txt+'\n\n'
# 	print(tweet['id_str']+'\n'+tweet['text']+ '\n\n\n')