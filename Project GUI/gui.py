import tkinter as tk
from tkinter import ttk
from Tkinter import *
import functions as f
import crypto_train as ct

app = tk.Tk()
app.title('ICT1002')
app.configure(background = "gray91")
# app.geometry('500x280')

# tabs
tab_parent = ttk.Notebook(app)

scraping_tab = ttk.Frame(tab_parent)
data_tab = ttk.Frame(tab_parent)
twitter_tab = ttk.Frame(tab_parent)
bitcoin_tab = ttk.Frame(tab_parent)

tab_parent.add(scraping_tab, text="Data Scraping")
tab_parent.add(data_tab, text="Data Analysis")
tab_parent.add(twitter_tab, text="Twitter Analysis")
tab_parent.add(bitcoin_tab, text="Bitcoin Analysis")

tab_parent.pack()

# Data Scraping Tab #####################################################################################################
space0 =  tk.Label(scraping_tab, text="                                       ", background = "gray91")
space1 =  tk.Label(scraping_tab, text="                                       ", background = "gray91")
space2 =  tk.Label(scraping_tab, text="                                       ", background = "gray91")
space3 =  tk.Label(scraping_tab, text="                                       ", background = "gray91")
space4 =  tk.Label(scraping_tab, text="                                       ", background = "gray91")
title = tk.Label(scraping_tab, text="Data Scraping", font=("",20), background = "gray91")
filenameLabel = tk.Label(scraping_tab, text="Filename:", font=("", 15), background = "gray91")
fileName = tk.Entry(scraping_tab, width=52, highlightbackground = "gray91")
urlLabel = tk.Label(scraping_tab, text="URL:", font=("", 15), background = "gray91")
url = tk.Entry(scraping_tab, width=52, highlightbackground = "gray91")
scrapBtn = tk.Button(scraping_tab, text="Generate", highlightbackground = "gray91")

# Widgets placements
space0.grid(row=10, column=0)
space1.grid(row=10, column=1)
space2.grid(row=10, column=2)
space3.grid(row=10, column=3)
space4.grid(row=10, column=4)
title.grid(row=0, column=0, columnspan=5)
filenameLabel.grid(row=1, column=0, sticky='e')
fileName.grid(row=1, column=1, columnspan=3)
urlLabel.grid(row=3, column=0, sticky='e')
url.grid(row=3, column=1,columnspan=3)
scrapBtn.grid(row=4, column=0, columnspan=5)

# Data Analysis Tab ########################################################################################################

space0 =  tk.Label(data_tab, text="                                       ", background = "gray91")
space1 =  tk.Label(data_tab, text="                                       ", background = "gray91")
space2 =  tk.Label(data_tab, text="                                       ", background = "gray91")
space3 =  tk.Label(data_tab, text="                                       ", background = "gray91")
space4 =  tk.Label(data_tab, text="                                       ", background = "gray91")
title = tk.Label(data_tab, text="Data Analysis", font=("",20), background = "gray91")

# Widgets placements
space0.grid(row=10, column=0)
space1.grid(row=10, column=1)
space2.grid(row=10, column=2)
space3.grid(row=10, column=3)
space4.grid(row=10, column=4)
title.grid(row=0, column=0, columnspan=5)

# Twitter Analysis Tab #####################################################################################################
space0 =  tk.Label(twitter_tab, text="                                       ", background = "gray91")
space1 =  tk.Label(twitter_tab, text="                                       ", background = "gray91")
space2 =  tk.Label(twitter_tab, text="                                       ", background = "gray91")
space3 =  tk.Label(twitter_tab, text="                                       ", background = "gray91")
space4 =  tk.Label(twitter_tab, text="                                       ", background = "gray91")
title = tk.Label(twitter_tab, text="Twitter Analysis", font=("",20), background = "gray91")
keywordLabel = tk.Label(twitter_tab, text="Keyword/Hashtag:", background = "gray91")
keywordEntry = tk.Entry(twitter_tab, width=16, highlightbackground = "gray91")
numOftwitsLabel = tk.Label(twitter_tab, text="Amount to Analyze:", background = "gray91")
numOftwitsEntry = tk.Entry(twitter_tab, width=16, highlightbackground = "gray91")
analyzeTwitBtn = tk.Button(twitter_tab, text="Generate", highlightbackground = "gray91", command=lambda : f.twit_senti(keywordEntry, numOftwitsEntry, twitter_tab))

# Widgets placements
space0.grid(row=10, column=0)
space1.grid(row=10, column=1)
space2.grid(row=10, column=2)
space3.grid(row=10, column=3)
space4.grid(row=10, column=4)
title.grid(row=0, column=0, columnspan=5)
keywordLabel.grid(row=1, column=0, sticky='e')
keywordEntry.grid(row=1, column=1)
numOftwitsLabel.grid(row=2, column=0, sticky='e')
numOftwitsEntry.grid(row=2, column=1)
analyzeTwitBtn.grid(row=3, column=1)


# BitCoin Analysis Tab #####################################################################################################
path = tk.StringVar()

space0 =  tk.Label(bitcoin_tab, text="                                       ", background = "gray91")
space1 =  tk.Label(bitcoin_tab, text="                                       ", background = "gray91")
space2 =  tk.Label(bitcoin_tab, text="                                       ", background = "gray91")
space3 =  tk.Label(bitcoin_tab, text="                                       ", background = "gray91")
space4 =  tk.Label(bitcoin_tab, text="                                       ", background = "gray91")
title = tk.Label(bitcoin_tab, text="Bitcoin Analysis", font=("",20), background = "gray91")
filePath = tk.Entry(bitcoin_tab, textvariable=path, width=52, highlightbackground = "gray91")
browseBtn = tk.Button(bitcoin_tab, text="Browse", command = lambda: f.getCSV_path(path), highlightbackground = "gray91")
addtolistBtn = tk.Button(bitcoin_tab, text="Add to List", command = lambda: f.addpathtolist(pathListbox, path.get()), highlightbackground = "gray91")
pathListbox = tk.Listbox(bitcoin_tab, width=52, height=15, font=("", 12), highlightbackground = "gray91")
trainBtn = tk.Button(bitcoin_tab, text="Train Data", command = ct.trainData(), highlightbackground = "gray91")
buildBtn = tk.Button(bitcoin_tab, text="Build Model", command = lambda: f.getselectedValue(pathListbox, pathListbox.curselection()), highlightbackground = "gray91")
removelistitemBtn = tk.Button(bitcoin_tab, text="Remove", command = lambda: f.removeSelected(pathListbox, pathListbox.curselection()), highlightbackground = "gray91")

# Widgets placements
space0.grid(row=10, column=0)
space1.grid(row=10, column=1)
space2.grid(row=10, column=2)
space3.grid(row=10, column=3)
space4.grid(row=10, column=4)
title.grid(row=0, column=0, columnspan=5)
filePath.grid(row=1, column=1, columnspan=3)
browseBtn.grid(row=1, column=4, sticky='w')
addtolistBtn.grid(row=2, column=0, columnspan=5)
pathListbox.grid(row=3, column=1, columnspan=3)
buildBtn.grid(row=4, column=1)
trainBtn.grid(row=4, column=2)
removelistitemBtn.grid(row=4, column=3)

app.mainloop()