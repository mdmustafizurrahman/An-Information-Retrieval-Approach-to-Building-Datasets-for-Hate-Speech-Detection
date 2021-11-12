import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy

import json
import re
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from nltk import agreement
from sklearn.metrics import cohen_kappa_score
import glob
import copy
import os
from collections import OrderedDict
from nltk.stem import PorterStemmer
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import simpledorff

from both_dataset_plots import fleiss_kappa_v1, my_gwet_AC1_v1, my_cohen_kappa_v1


plot_location = "/home/nahid/Downloads/data_all/git_data/"
df_list = []


def if_terms_in_rationale(tweet, highlighted_terms1, highlighted_terms2, highlighted_terms3):
    global df_list
    annotator1_highlighted_terms = []
    annotator2_highlighted_terms = []
    annotator3_highlighted_terms = []

    # print(tweet)

    all_terms = []
    a = None

    if highlighted_terms1 != "Nothing":
        for terms in highlighted_terms1.split(";"):
            all_terms = terms.split(" ")
            for a in all_terms:
                annotator1_highlighted_terms.append(a)
    all_terms = []
    a = None
    if highlighted_terms2 != "Nothing":
        for terms in highlighted_terms2.split(";"):
            all_terms = terms.split(" ")
            for b in all_terms:
                annotator2_highlighted_terms.append(a)

    all_terms = []
    a = None
    if highlighted_terms3 != "Nothing":
        for terms in highlighted_terms3.split(";"):
            all_terms = terms.split(" ")
            for a in all_terms:
                annotator3_highlighted_terms.append(a)

    # print(annotator1_highlighted_terms, annotator2_highlighted_terms, annotator3_highlighted_terms)

    for terms in tweet.split(" "):
        term_list = []
        term_list.append(terms)
        if terms in annotator1_highlighted_terms:
            term_list.append(1)
        else:
            term_list.append(0)

        if terms in annotator2_highlighted_terms:
            term_list.append(1)
        else:
            term_list.append(0)

        if terms in annotator3_highlighted_terms:
            term_list.append(1)
        else:
            term_list.append(0)

        df_list.append(copy.deepcopy(term_list))



def main():
    global df_list
    df = pd.read_csv("/home/nahid/Downloads/data_all/git_data/with_tweet-ids/all_data_with_everything/main_subset_everything.csv")
    # dropping unnecessary columns
    df = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    df = df.fillna("Nothing")

    # iterating the columns
    for col in df.columns:
        print(col)

    for index, row in df.iterrows():
        tweet = row['tweet']


        # highlighted terms are the contrained ratinales from workers
        # terms are separated by ;
        highlighted_terms1 = row['highlighted_terms1']
        highlighted_terms2 = row['highlighted_terms2']
        highlighted_terms3 = row['highlighted_terms3']

        # print(index, highlighted_terms1, highlighted_terms2, highlighted_terms3)

        if_terms_in_rationale(tweet, highlighted_terms1, highlighted_terms2, highlighted_terms3)


    # print(df_list)

    termwise_annotator_labels = pd.DataFrame(df_list, columns=['term', 'annotator_1','annotator_2','annotator_3',])

    print(termwise_annotator_labels)

    rater1 = termwise_annotator_labels['annotator_1']
    rater2 = termwise_annotator_labels['annotator_2']
    rater3 = termwise_annotator_labels['annotator_3']
    taskdata = [[0, str(i), str(rater1[i])] for i in range(0, len(rater1))] + [[1, str(i), str(rater2[i])] for i in
                                                                               range(0, len(rater2))] + [
                   [2, str(i), str(rater3[i])] for i in range(0, len(rater3))]

    # print(taskdata)

    ratingtask = agreement.AnnotationTask(data=taskdata)
    print("fleiss " + str(ratingtask.multi_kappa()))
    print("alpha " + str(ratingtask.alpha()))

    rater1 = termwise_annotator_labels['annotator_1'].tolist()
    rater2 = termwise_annotator_labels['annotator_2'].tolist()
    rater3 = termwise_annotator_labels['annotator_3'].tolist()

    kappa12 = my_cohen_kappa_v1(rater1, rater2, rater3, 1, 2)
    kappa13 = my_cohen_kappa_v1(rater1, rater2, rater3, 1, 3)
    kappa23 = my_cohen_kappa_v1(rater1, rater2, rater3, 2, 3)

    print("aveargae kappas:", (kappa12 + kappa13 + kappa23) / 3.0)

    print("Fleiss Kappas:", fleiss_kappa_v1(rater1, rater2, rater3))

    AC12 = my_gwet_AC1_v1(rater1, rater2, rater3, first=1, second=2)
    AC13 = my_gwet_AC1_v1(rater1, rater2, rater3, first=1, second=3)
    AC23 = my_gwet_AC1_v1(rater1, rater2, rater3, first=2, second=3)

    print("aveargae GWET AC1:", (AC12 + AC13 + AC23) / 3.0)

if __name__ == "__main__":
    main()