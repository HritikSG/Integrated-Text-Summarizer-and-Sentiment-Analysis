#The following code merges two different datasets together as dataframe for finetuning of any text summarizer model or algorithm
#The datasets are BBC News Articles and Amazon Reviews and their summaries.
#In total it's around 330.000 lines of texts and summaries.
#The program outputs a csv file for training.
#The code is not structured well and it's due to its one-off usage.
#It is written by Ramazan Mengi on October 2023 in University of Michigan at Dearborn.

import os
import glob as glob
from io import open
from tensorflow.keras.layers import *
import warnings
import numpy as np
import pandas as pd
from preprocessor import text_cleaner

articles_path = "/Users/ramazanmengi/PycharmProjects/KerasT5TextSummarizerFineTuning/Datasets/BBC News Summary/News Articles"
summaries_path = "/Users/ramazanmengi/PycharmProjects/KerasT5TextSummarizerFineTuning/Datasets/BBC News Summary/Summaries"

categories_list = os.listdir(articles_path)


def read_file(articles_path, summaries_path, categories_list, encoding="ISO-8859-1"):
    articles = []
    summaries = []
    categories = []
    for category in categories_list:
        article_paths = glob.glob(os.path.join(articles_path, category, '*.txt'), recursive=True)
        summary_paths = glob.glob(os.path.join(summaries_path, category, '*.txt'), recursive=True)

        print(
            f'found {len(article_paths)} file in articles/{category} folder, {len(summary_paths)} file in summaries/{category}')

        if len(article_paths) != len(summary_paths):
            print("number of files is not equal")
            return
        for file in range(len(article_paths)):
            categories.append(category)
            with open(article_paths[file], mode='r', encoding=encoding) as files:
                articles.append(files.read())

            with open(summary_paths[file], mode='r', encoding=encoding) as files:
                summaries.append(files.read())

    print(f'total {len(articles)} file in articles folder and {len(summaries)} files in summaries folder')
    return articles, summaries

articles, summaries = read_file(articles_path, summaries_path, categories_list)

df = pd.DataFrame({'text' :articles, 'summary' :summaries},)

df = df.dropna()



print(df)
cleaned_texts = []

for t in df["text"]:
    cleaned_texts.append(text_cleaner(t, 0))

# clean summaries
cleaned_summaries = []

for s in df["summary"]:
    cleaned_summaries.append(text_cleaner(s, 1))

# create new dataframe for the cleaned texts and summaries
dataframe2 = pd.DataFrame(columns=["text", "summary"])
# put cleaned version of texts and summaries into the dataframe
dataframe2["text"] = cleaned_texts
dataframe2["summary"] = cleaned_summaries

dataframe2.replace('', np.nan, inplace=True)
dataframe2.dropna(axis=0,inplace=True)

print(dataframe2)

df.to_csv('texttosummarydataset2.csv', index=False)
dataframe2.to_csv('texttosummarydataset2.csv', index=False)








#
# pd.set_option("display.max_colwidth", 200)
# warnings.filterwarnings("ignore")
#
#
# # load the data
# df2 = pd.read_csv("/Users/ramazanmengi/PycharmProjects/KerasT5TextSummarizerFineTuning/Datasets/Reviews.csv")
# df2.head()
#
#
# # Drop Duplicates and NAs
# df2.drop_duplicates(subset=['Text'],inplace=True)
# df2.dropna(axis=0,inplace=True)
#
# # clean the text
# cleaned_texts = []
#
# for t in df2["Text"]:
#     cleaned_texts.append(text_cleaner(t, 0))
#
# # clean summaries
# cleaned_summaries = []
#
# for s in df2["Summary"]:
#     cleaned_summaries.append(text_cleaner(s, 1))
#
# # create new dataframe for the cleaned texts and summaries
# dataframe = pd.DataFrame(columns=["text", "summary"])
# # put cleaned version of texts and summaries into the dataframe
# dataframe["text"] = cleaned_texts
# dataframe["summary"] = cleaned_summaries
#
# # dataframe.replace('', np.nan, inplace=True)
# # dataframe.dropna(axis=0,inplace=True)
# #
# # print(dataframe)
#
# # clean the text
# cleaned_texts = []
#
#
#
# frames = [dataframe, dataframe2]
#
# result = pd.concat(frames)
#
# # Reset the index to create a new continuous index
#
# result = result.iloc[np.random.permutation(len(result))].reset_index(drop=True)
# print(result)
# result.to_csv('texttosummarydataset.csv', index=False)
