import csv
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from textblob import TextBlob
import tweepy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import numpy as np

# get csv path and set to show in textbox
def getCSV_path(textbox):
    import_file_path = filedialog.askopenfilename()
    textbox.set(import_file_path)

# get listbox and path then add path into listbox
def addpathtolist(listbox, path):
    listbox.insert('end', path)

# removes selected items from listbox
def removeSelected(listbox, selected):
    listbox.delete(selected, 'anchor')

# get value of selected file path from listbox
def getselectedValue(listbox, selected):
    print listbox.get(selected)

def percent(value, max):
    return 100 * float(value) / float(max)

def twit_senti(word, num, frame):
    consumerKey = 'F22OIZyHOeZsFslqeaBMRJpoq'
    consumerSecret = 'ZkQQML2i4BC2c9pp10B7ZzSq39ShA2YqsdinSMOUBMvu8cT4vp'
    accessToken = '223413298-lFZ1qAOM8TM4I2y9aSGQO5v6QCEJ0onaXs3FQdWr'
    accessTokenSecret = 'SOhhQKdE6vvTkazK5dOYwSNBgwzM4J38yrNsSfKi9h24U'

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)

    searchTerm = word.get()
    noOfSearchTerms = int(num.get())

    tweets = tweepy.Cursor(api.search, q=searchTerm).items(noOfSearchTerms)

    positive = 0
    negative = 0
    neutral = 0
    polarity = 0

    for tweet in tweets:
        analysis = TextBlob(tweet.text)
        polarity += analysis.sentiment.polarity

        if (analysis.sentiment.polarity == 0):
            neutral += 1
        elif (analysis.sentiment.polarity < 0.00):
            negative += 1
        elif (analysis.sentiment.polarity > 0.00):
            positive += 1
    total = positive + neutral + negative
    positive = format(percent(positive, total), '.2f')
    negative = format(percent(negative, total), '.2f')
    neutral = format(percent(neutral, total), '.2f')

    labels = ['Positive[' + str(positive) + '%]', 'Neutral[' + str(neutral) + '%]', 'Negative[' + str(negative) + '%]']
    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'gold', 'red']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.legend(patches, labels, loc='best')
    plt.title('Reactions on ' + searchTerm + ' analyzed with ' + str(total) + 'out of' + str(noOfSearchTerms) + 'Tweets.')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

