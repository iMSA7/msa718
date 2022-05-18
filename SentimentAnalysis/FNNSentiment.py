"""
Created on Thu Jan 28 00:01:06 2021

@author: muhannad
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
import os
import pandas as pd
import pandas_datareader as web
import datetime
import SentimentAnalysis
import TwitterAPI


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


def get_data_paths(stock="AAPL", training_percentage=0.8):
    paths = glob(os.path.join('SentimentAnalysis', 'Data', f'{stock}_training', '*'))
    training_paths = paths
    num_of_training_paths = int(len(paths) * training_percentage)
    validation_paths = []
    training_paths = paths[:num_of_training_paths]
    validation_paths = paths[num_of_training_paths:]
    return training_paths, validation_paths


class TrainDataset(data.Dataset):
    def __init__(self, paths):
        super(TrainDataset, self).__init__()
        self.paths = paths

        data_list = []
        date_list = []

        for path in self.paths:
            data_list.append(pd.read_csv(path, sep='\t', names=['neg', 'neu', 'pos', 'currPrev', 'curr20',
                                                                'next1', 'next2', 'next3', 'next4', 'next5']))
            date_list.append(path[-10:])

        self.all_data = pd.concat(data_list)
        self.all_data.index = date_list
        self.x = np.array(self.all_data.loc[:, ['neg', 'neu', 'pos', 'currPrev', 'curr20']].values.tolist())
        self.y = np.array(self.all_data.loc[:, ['next1', 'next2', 'next3', 'next4', 'next5']].values.tolist())

    def __getitem__(self, index):
        example = self.x[index]
        output = self.y[index]
        return torch.from_numpy(example).float(), torch.from_numpy(output).float()

    def __len__(self):
        return len(self.paths)


class TestDataset(data.Dataset):
    def __init__(self, paths):
        super(TestDataset, self).__init__()
        self.paths = paths

        data_list = []
        date_list = []

        for path in self.paths:
            data_list.append(pd.read_csv(path, sep='\t', names=['neg', 'neu', 'pos', 'currPrev', 'curr20',
                                                                'next1', 'next2', 'next3', 'next4', 'next5']))
            date_list.append(path[-10:])

        self.all_data = pd.concat(data_list)
        self.all_data.index = date_list
        self.x = np.array(self.all_data.loc[:, ['neg', 'neu', 'pos', 'currPrev', 'curr20']].values.tolist())
        self.y = np.array(self.all_data.loc[:, ['next1', 'next2', 'next3', 'next4', 'next5']].values.tolist())

    def __getitem__(self, index):
        example = self.x[index]
        output = self.y[index]
        return torch.from_numpy(example).float(), torch.from_numpy(output).float()

    def __len__(self):
        return len(self.paths)


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_size = 5
        self.hidden_size = 5
        self.num_classes = 5
        self.l1_2 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.l2_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3_4 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        out = self.l1_2(x)
        out = self.relu(out)
        out = self.l2_3(out)
        out = self.relu(out)
        return self.l3_4(out)


def main(stock_symbol, train):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 20
    learning_rate = 1e-2
    model = NeuralNetwork().to(device)

    # Loss = nn.MSELoss()
    Loss = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.20)

    # Training
    saved_model_path = 'SentimentAnalysis/sentModel.sav'
    training_percent = 0.8

    if train == 'yes':
        training_paths, validation_paths = get_data_paths(training_percentage=training_percent)
        # training parms
        min_loss = float('inf')
        train_losses = []
        num_workers = 0
        batch_size = 20
        train_set = TrainDataset(training_paths)
        training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)
        # val parms
        val_losses = []
        val_set = TrainDataset(validation_paths)
        validation_data_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)

        # start training
        model.train()
        for epoch in range(epochs):
            for iteration, sample in enumerate(training_data_loader):
                x, y = sample
                print("Epochs " + str(epoch + 1), "Iteration " + str(iteration + 1))
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                pred = model(x)

                loss = Loss(pred.squeeze(), y)
                train_losses.append(loss.item())
                print('Loss: ', loss.item(), " minimum: ", min(train_losses), end=" | ")

                # backward
                loss.backward()
                optimizer.step()

                # evaluation
                with torch.no_grad():
                    model.eval()
                    temp = []
                    for i, val_sample in enumerate(validation_data_loader):
                        x_val, y_val = val_sample
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)
                        pred = model(x_val)
                        val_loss = Loss(pred.squeeze(), y_val)
                        temp.append(val_loss.item())

                    average = sum(temp) / len(temp)
                    val_losses.append(average)
                    print('Test Loss: ', average, " minimum: ", min(val_losses))
                    if average < min_loss:
                        min_loss = average
                        torch.save(model.state_dict(), saved_model_path)

        training_parameters = f"Sentiment analysis model training.\nNeural network size: [5, 5, 5, 5].\nTotal Days: " \
                              f"{len(training_paths) + len(validation_paths)}.\nTraining percentage: " \
                              f"{training_percent * 100}%.\nLoss function: {Loss.__str__()[:-2]}.\nEpochs: {epochs}," \
                              f" Batch Size: {batch_size}."
        with open("Training_Sentiment", 'w') as f:
            f.write(training_parameters)

        train_losses = np.array(train_losses) * 100
        val_losses = np.array(val_losses) * 100

        plt.figure()
        # plt.plot(range(len(train_losses)), train_losses, '-g', label="Training loss")
        plt.plot(range(len(val_losses)), val_losses, '-r', label="Validation loss")
        plt.title("Losses")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('st.png', bbox_inches='tight')
    else:
        model.load_state_dict(torch.load(saved_model_path))
        model.eval()
        testing = False
        # test or predict?
        if not testing:
            # predict
            start_date = convert_date(str(datetime.datetime.now())[:10], -50)  # 50 days in the past
            end_date = str(datetime.datetime.now())[:10]  # today's date
            data = web.DataReader(stock_symbol, data_source='yahoo', start=start_date,
                                  end=end_date)

            prices = list(data.Close)
            last_20 = prices[-21:-1]
            dates = [str(x)[:10] for x in data.index]
            client = TwitterAPI.get_client(TwitterAPI.authenticate())
            status = TwitterAPI.search_tweets(client, stock_symbol, count=100, search_type='popular')

            time, tweets = TwitterAPI.process_status(status)
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

            current_price = prices[-1]
            previous_price = prices[-2]
            current_over_prev = current_price / previous_price
            last_20_average = sum(last_20) / len(last_20)
            curr_over_20 = current_price / last_20_average

            in_data = np.array([neg - 0.5, neutral - 0.5, pos - 0.5, current_over_prev - 1, curr_over_20 - 1]).astype(np.float32)
            prediction = model(torch.from_numpy(in_data)).detach().numpy()
            current = prices[-1]
            prediction = prediction * current
            # plotting settings
            for day in prediction:
                dates.append(convert_date(dates[-1], 1))
            days_to_show = -10
            plt.figure()
            line = plt.subplot(1, 1, 1)
            color = '.-g' if sum(prediction) / len(prediction) > current else '.-r'
            line.plot(range(-len(prices[days_to_show:]) + 1, len(prediction) + 1)[-len(prediction):], prediction, color,
                      label="prediction")
            line.plot(range(-len(prices[days_to_show:]) + 1, len(prediction) + 1)[:len(prices[days_to_show:])],
                      prices[days_to_show:], '.-b', label="True")

            line.set_xticks(range(-9, 6))
            line.set_xticklabels([x[-5:] for x in dates[-15:]], rotation='vertical', fontsize=8)
            line.grid(True, which='both')
        else:
            training_paths, validation_paths = get_data_paths(stock_symbol, training_percentage=training_percent)
            training_paths.sort()
            training_paths = training_paths[50:255]
            print(training_paths[0])
            print(training_paths[-1])
            train_set = TrainDataset(training_paths)
            training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=1,
                                              shuffle=False)
            dates = [x[-10:] for x in training_paths]
            start_date = dates[0]
            end_date = dates[-1]
            data = web.DataReader(stock_symbol, data_source='yahoo', start=start_date,
                                  end=end_date)

            dates = [str(x)[:10] for x in data.index]
            prices = list(data.Close)
            predictions = []
            for i, sample in enumerate(training_data_loader):
                if i % 5 == 0 and i < len(prices):
                    x, y = sample
                    prediction = model(x).detach().numpy() * prices[i]
                    predictions = predictions + prediction.tolist()[0]

            plt.figure()
            line = plt.subplot(1, 1, 1)
            print(len(predictions))
            print(len(prices))
            line.plot(range(len(predictions)), predictions, '.-r',
                      label="prediction")
            line.plot(range(len(prices)),
                      prices, '.-b', label="True")
            line.set_xticks(range(len(prices)))
            line.set_xticklabels([x[-5:] for x in dates], rotation='vertical', fontsize=2)
            line.grid(False, which='both')

        plt.title(f"Price of {stock_symbol}")
        plt.xlabel("Timeframe (Days)")
        plt.ylabel("price (USD)")
        plt.legend(loc='best')
        plt.savefig('sentiment.png', bbox_inches='tight')


if __name__ == '__main__':
    import sys
    main(sys.argv[1:][0], sys.argv[1:][1])
