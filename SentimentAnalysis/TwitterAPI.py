import re
from tweepy import API
from tweepy import OAuthHandler
import pandas as pd
import pandas_datareader as web
import datetime
import csv
import DataManager

from SecretKeys import *


def authenticate():
    """
    needed for an authenticated access to Twitter API
    :return: tweepy.OAuthHandler
    """
    try:
        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET_KEY)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth
    except:
        print('authentication Error')


def get_tweets_from_user(client, userID, count=1, tweet_mode='extended'):
    """
    retrieves tweets from a specific user timeline.;
    :param client: authenticated tweepy.API
    :param userID: str of 'user ID'
    :param count: number of tweets
    :param tweet_mode: extended
    :return: status list (of tweets)
    """
    tweets = client.user_timeline(screen_name=userID, count=count, tweet_mode=tweet_mode)
    return tweets


def search_tweets(client, keyword, tweet_mode='extended', count=1, lang='en', search_type='mixed',
                  start_date=(str(datetime.datetime.now() - datetime.timedelta(days=7))[:10]),
                  end_date=str(datetime.datetime.now() - datetime.timedelta(days=1))[:10]):
    """
    searches for tweets given search query
    :param client: tweepy.API
    :param keyword: search query
    :param tweet_mode: extended
    :param count: number of tweets (max=10O)
    :param lang: language eg. 'en'
    :param search_type: recent, popular or mixed
    :param start_date: of search scope (max = 7 days in the past)
    :param end_date: of search scope
    :return: list of status each containing tweets and more details.
    """
    tweets = client.search(q=keyword, tweet_mode=tweet_mode,
                           count=count, lang=lang, since=start_date, until=end_date, result_type=search_type)
    return tweets


def write_to_file(data, file):
    """
    writes data into csv file
    :param data: list of tuples e.g: zip(list1, list2)
    :param file: str of file path/name
    """
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def process_status(status):
    """
    takes a list of status and returns a tuple of two lists (str-list=time, str-list=tweet)
    :param status: list of status
    :return: tuple of string lists (time-stamps, tweets)
    """
    time = []
    tweets = []
    for tweet in status:
        if hasattr(tweet, 'retweeted_status'):
            try:
                text = tweet.retweeted_status.full_text
                time.append(str(tweet.created_at)[:10])
                tweets.append(text)
            except:
                print('error: Re-tweet attribute')
        else:
            try:
                text = tweet.full_text
                time.append(str(tweet.created_at)[:10])
                tweets.append(text)
            except AttributeError:
                print('error Tweet attribute')

    return time, tweets


def get_stock_data(stock='AAPL', start_date=datetime.datetime.now() - datetime.timedelta(days=10),
                   end_date=datetime.datetime.now()):
    data = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)
    return data


def convert_date(text_date, int_diff):
    """
        gets a date that is different from 'text_date' by 'int_diff'
        :param text_date: string date in YYYY-MM-DD format
        :param int_diff: days different from the given date, negative integers return date in the past.
        :return: Date class
        >>> convert_date('2020-09-07', -3) == '2020-09-04'
        True
        >>> convert_date('2020-09-07', 3) == '2020-09-10'
        True
        >>> convert_date('2020-09-07', 0) == '2020-09-07'
        True
        >>> convert_date('2020-09-07', -3) == '2020-09-07'
        False
        >>> convert_date('2020-09-07', -3) == '2020-09-10'
        False
        >>> convert_date('2020-09-07', 3) == '2020-09-04'
        False

        """
    diff = abs(int_diff)
    date = datetime.datetime.strptime(text_date, '%Y-%m-%d')
    if int_diff > 0:
        date = date + datetime.timedelta(days=diff)
        return str(date)[:10]
    elif int_diff < 0:
        date = date - datetime.timedelta(days=diff)
        return str(date)[:10]
    else:
        return text_date


def get_next_timestamp_price(time, data):
    next_timestamps = []
    for date in time:
        next = convert_date(date, 1)
        while next not in data.index:
            next = convert_date(next, 1)
        next_timestamps.append(data.loc[next]['Close'])

    return next_timestamps


def get_client(authentication):
    return API(authentication, wait_on_rate_limit=True)


def main():
    client = get_client(authenticate())

    stock = '$AAPL'
    start_date = '2020-10-28'  # format should be 'YYYY-MM-DD'
    start_date = str(datetime.datetime.now() - datetime.timedelta(days=7))[:10]
    delta = 7  # days different from start day
    end_date = convert_date(start_date, delta)

    status = search_tweets(client, stock, count=100, search_type='mixed')

    time, tweets = process_status(status)

    df = pd.DataFrame(zip(time, tweets), columns=['Date', 'Tweet'])

    for tweet in list(df.Tweet):
        print("===================ORIGINAL=====================")
        print(tweet)
        print("===================FILTERED=====================")
        print(DataManager.clear_text(tweet))

    print(df)


if __name__ == '__main__':
    main()
