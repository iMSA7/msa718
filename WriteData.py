import numpy as np


def convert_date(text_date, int_diff):
    """
        get a date that is different from 'text_date' by 'int_diff'
        :param text_date: string date in YYYY-MM-DD format
        :param int_diff: days different from the given date, negative integers return date in the past.
        :return: Date String of the input format
    """
    import datetime
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


def write_data_sentiment(stock_symbol, start_date, end_date):
    """
    reads tweets from files of a stock writes them in a new folder in this format: '<stock symbol/dates>'.
    search is by looking for the stock symbol folder and date files inside.
    e.g: AAPL/2017-01-19
    :param stock_symbol: stock symbol e.g: AAPL for Apple stock
    :param start_date: string date in YYYY-MM-DD format
    :param end_date: string date in YYYY-MM-DD format
    :return: writes files in '<stock symbol>' folder which contains the data of each date
    """
    from SentimentAnalysis import SentimentAnalysis
    import json
    import pandas_datareader as web

    stock_data = web.DataReader(stock_symbol, data_source='yahoo',
                                start=convert_date(start_date, -50),
                                end=convert_date(end_date, 14))
    dates_list = stock_data.index
    path_to_tweets = f"SentimentAnalysis/Data/{stock_symbol}/"

    current_date = start_date

    while current_date != end_date:
        # read from file and assign the data to a panda's dataframe:
        in_file = path_to_tweets + current_date
        tweets = []
        try:
            with open(in_file, 'r') as f:
                for line in f:
                    tweets.append(json.loads(line))
        except FileNotFoundError:
            print(f"Error: file \"{in_file}\" not found ")
            current_date = str(convert_date(current_date, 1))[:10]
            continue

        tweets = set([x['text'] for x in tweets])
        tweets = list(tweets)
        stemmed_rev = [SentimentAnalysis.DataManager.stemmer_stop(SentimentAnalysis.DataManager.clear_text(rev))
                       for rev in tweets]

        rev_vector = SentimentAnalysis.vectorizor.transform(stemmed_rev)

        # predictions
        prediction_NB = SentimentAnalysis.clf_NB.predict(rev_vector)
        prediction_SVM = SentimentAnalysis.clf_SVM.predict(rev_vector)
        prediction_FT = SentimentAnalysis.clf_FT.predict(rev_vector)

        # majority-vote-based labelling
        labels = [SentimentAnalysis.mode((a, b, c))
                  for a, b, c, in zip(prediction_NB, prediction_SVM, prediction_FT)]

        neg = labels.count(-1) / len(labels)
        neutral = labels.count(0) / len(labels)
        pos = labels.count(1) / len(labels)

        i = 0
        temp_date = convert_date(current_date, i)
        while temp_date not in dates_list:
            i -= 1
            temp_date = convert_date(current_date, i)

        last_price_date = str(temp_date)[:10]
        current_price = stock_data.loc[convert_date(last_price_date, 0)]['Close']

        i = -1
        temp_date = convert_date(last_price_date, i)
        while temp_date not in dates_list:
            i -= 1
            temp_date = convert_date(last_price_date, i)

        current_over_prev = current_price / stock_data.loc[temp_date]['Close']
        # past 20 days closing price
        i = -1
        count = 0
        closing_prices = []
        temp_date = convert_date(last_price_date, i)
        while count < 20:
            if temp_date in dates_list:
                count += 1
                closing_prices.append(stock_data.loc[temp_date]['Close'])

            i -= 1
            temp_date = convert_date(last_price_date, i)

        current_over_20 = current_price / (sum(closing_prices) / len(closing_prices))

        # get the next five closing prices
        i = 1
        count = 0
        next_five = []
        temp_date = convert_date(last_price_date, i)
        while count < 5:
            if temp_date in dates_list:
                count += 1
                next_five.append(stock_data.loc[temp_date]['Close'] / current_price)

            i += 1
            temp_date = convert_date(last_price_date, i)

        # writing to file
        path_to_write = f"SentimentAnalysis/Data/{stock_symbol}_training/"
        # standardise data between [0.5, -0.5]
        data_to_write = [neg - 0.5, neutral - 0.5, pos - 0.5, current_over_prev - 1, current_over_20 - 1] + next_five
        out_file = path_to_write + current_date

        with open(out_file, 'w') as f:
            f.write('\t'.join([str(x) for x in data_to_write]))
            print(f"file \"{out_file}\" has been completed!")

        # update the current date
        current_date = str(convert_date(current_date, 1))[:10]

    return str(convert_date(current_date, 0))[:10]


def get_next_five(prices):
    list_of_fives = []
    for i in range(len(prices) - 5):
        cut = np.array(prices[i + 1:i + 6]) / prices[i]
        list_of_fives.append(cut)
    return list_of_fives


def write_data_technical(stock_symbol, start_date, end_date):
    """
    write the technical data into files.
    :param stock_symbol: the stock symbol in capitals.
    :param start_date: starting date in YYYY-MM-DD format.
    :param end_date: until date in YYYY-MM-DD format.
    :return: return the name of the last written file.
    """
    import pandas_datareader as web
    import TechnicalAnalysis.TechnicalIndicators as Technical_indicators

    data = web.DataReader(stock_symbol, data_source='yahoo', start=start_date,
                          end=Technical_indicators.convert_date(end_date, 14))
    print(data)
    prices = list(data.Close[:-5])
    last_five = list(data.Close[-5:])
    highs = list(data.High)
    lows = list(data.Low)
    dates = [str(x)[:10] for x in data.index]
    moving_average200 = Technical_indicators.EMA(prices, period=200)
    moving_average50 = Technical_indicators.EMA(prices)
    average_true_range = Technical_indicators.ATR(prices, highs, lows)
    william = Technical_indicators.williams(prices, highs, lows)
    rates = Technical_indicators.ROC(prices)
    next_five = get_next_five(prices + last_five)
    # writing data.
    path_to_write = f"TechnicalAnalysis/{stock_symbol}_training/"
    for i in range(len(prices)):
        out_file = path_to_write + dates[i]
        data_to_write = [rates[i] - 1, william[i] - 0.5, (average_true_range[i] / prices[i]) - 0.5,
                         (moving_average50[i] / prices[i]) - 1, (moving_average200[i] / prices[i]) - 1] + list(
            next_five[i])

        with open(out_file, 'w') as f:
            f.write('\t'.join([str(x) for x in data_to_write]))
            print(f"file \"{out_file}\" has been completed!")

    return len(prices)


def main():
    choice = input("collect data from Sentiment (s) or technical (t) or both (b):\n")

    stock = input("Enter the stock symbol:\n")
    start_date = '2014-01-01'
    end_date = '2020-01-01'
    if choice == "b":
        stopped_at = write_data_sentiment(stock, start_date, end_date)
        print(f"the data written until {stopped_at}")
        days = write_data_technical(stock, start_date, end_date)
        print(f"the data written: {days}")
    elif choice == 's':
        stopped_at = write_data_sentiment(stock, start_date, end_date)
        print(f"the data written until {stopped_at}")
    else:
        days = write_data_technical(stock, start_date, end_date)
        print(f"the data written: {days} days")


if __name__ == '__main__':
    main()
