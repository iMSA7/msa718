import datetime


def convert_date(text_date, int_diff):
    """
        get a date that is different from 'text_date' by 'int_diff'
        :param text_date: string date in YYYY-MM-DD format
        :param int_diff: days different from the given date, negative integers return date in the past.
        :return: Date String of the input format
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


def EMA(prices, period=50):
    """
    method for calculating the exponent moving average.
    :param prices: list of the prices of a stock.
    :param period: period of the moving average.
    :return: the moving average list.
    """
    scalar = 2 / (period + 1)
    moving_averages = []
    for i in range(len(prices)):
        if i == 0:
            if period >= len(prices):
                moving_averages.append(sum(prices) / len(prices))
            else:
                moving_averages.append(sum(prices[: period]) / period)
        else:
            MA = (prices[i] * scalar) + moving_averages[-1] * (1 - scalar)
            moving_averages.append(MA)

    return moving_averages


def ATR(prices, highs, lows, period=5):
    """
    Average true range indicator.
    :param prices: list of closing prices.
    :param highs: list of the highs.
    :param lows: list of the lows.
    :param period: integer representing days.
    :return: list of ATR in the length of prices.
    """
    ATRs = []
    TRs = []
    for i in range(len(prices)):
        if i == 0:
            TRs.append(highs[i] - lows[i])
            ATRs.append(TRs[i])
        else:
            TRs.append(max(highs[i] - lows[i], abs(highs[i] - prices[i - 1]), abs(lows[i] - prices[i - 1])))
            if len(TRs) > period:
                avt = (ATRs[i - 1] * (period - 1) + TRs[i]) / period
                ATRs.append(avt)
            else:
                ATRs.append(sum(TRs) / len(TRs))

    return ATRs


def williams(prices, highs, lows, period=14):
    """
    Williams rate indicator.
    :param prices: list of closing prices.
    :param highs: list of the highs.
    :param lows: list of the lows.
    :param period: integer representing days.
    :return: list of williams values in the length of prices.
    """
    wlls = []
    for i in range(len(prices)):
        current = prices[i]
        if i < period:
            if i > 0:
                highest = max(highs[:i+1])
                lowest = min(lows[:i+1])
            else:
                highest = highs[i]
                lowest = lows[i]
            wlls.append((highest - current) / (highest - lowest))
        else:
            highest = max(highs[i - period:i+1])
            lowest = min(lows[i - period:i+1])
            wlls.append((highest - current) / (highest - lowest))
    return wlls


def ROC(prices):
    """
    rate of change.
    :param prices: list of the closing prices.
    :return: list of the rates of change.
    """
    rates = []
    for i in range(len(prices)):
        if i == 0:
            rates.append(1.0)
        else:
            rates.append(prices[i] / prices[i - 1])
    return rates
