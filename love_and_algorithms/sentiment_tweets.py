#
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob

women_list = {
    'sifiso': {
        'twitter':'@Sifiso7',
        'facebook': 'Sifiso Chasafara'
    },
    'a': {
        'twitter':'@',
        'facebook': 'name_a'
    },
    'b': {
        'twitter':'@',
        'facebook': 'name_b'
    }
}
class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = 'X'
        consumer_secret = 'X'
        access_token = 'X'
        access_token_secret = 'X'
 
        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")
 
    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
 
    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        #Assign Polarity Threshold
        threshold = 0.4
        # create TextBlob object of passed tweet text
        sent_analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if sent_analysis.sentiment.polarity > threshold:
            return 'positive'
        elif sent_analysis.sentiment.polarity < -threshold:
            return 'negative'
        else:
            return 'neutral'
 
    def get_tweets(self, query, count = 10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []
 
        try:
            # call twitter api to fetch tweets
            
            ################# Search for tweets with 'q' #################
            # fetched_tweets = self.api.search(q = query, count = count)

            ################# Search for tweets by user #################            
            fetched_tweets = self.api.user_timeline(screen_name = 'Sifiso7',count = count)
 
            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}
 
                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
 
                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
 
            # return parsed tweets
            return tweets
 
        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
 
def main():
    # creating object of TwitterClient Class
    api = TwitterClient()
    # calling function to get tweets
    tweets = api.get_tweets(query = '@Sifiso7', count = 200)
 
    # picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    # percentage of positive tweets
    if len(ptweets)>0:
        print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    # percentage of negative tweets
    if len(ntweets)>0:
        print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
    # percentage of neutral tweets
    if (len(tweets) - len(ntweets) - len(ptweets)) >0:
        print("Neutral tweets percentage: {} % \
            ".format(100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets)))
 
    # printing first 5 positive tweets
    if len(ptweets) >= 5:
        print("\n\nPositive tweets [n={}]:".format(len(ptweets)))
        for tweet in ptweets[:5]:
            print(tweet['text'])
    else:
        print("\n\nPositive tweets [n={}]:".format(len(ptweets)))
        for tweet in ptweets:
            print(tweet['text'])
 
    # printing first 5 negative tweets
    if len(ntweets) >= 5:
        print("\n\nNegative tweets [n={}]:".format(len(ntweets)))
        for tweet in ntweets[:5]:
            print(tweet['text'])
    else:
        print("\n\nNegative tweets [n={}]:".format(len(ntweets)))
        for tweet in ntweets:
            print(tweet['text'])
 
if __name__ == "__main__":
    # calling main function
    main()
