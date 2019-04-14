# =============================================================
# Load in library
# =============================================================

import pandas as pd
import numpy as np
import glob
import enchant

# from textblob import TextBlob
# from langdetect import detect
# from langdetect.lang_detect_exception import LangDetectException

# ==============================================================
# EDA
# ==============================================================

# Read in data
# TODO: File 17 & 18 have issues and cannot be read in hence I put them into bad files folder will deal with them later.

all_files = glob.glob("Datasets/*.csv")

li = []

for file in all_files:
    print(file)
    df = pd.read_csv(file, error_bad_lines=False, encoding='utf8')
    li.append(df)

review_raw = pd.concat((pd.read_csv(file, error_bad_lines=False, encoding='utf8')
                        for file in all_files))


print(review_raw.shape)

# Check if all foreign languages have translation

review = review_raw.copy()

nan_body_rows = review[review['Body'].isnull()]

nan_sub_rows = review[review['Subject'].isnull()]

# Fill nan body with subject

review['Body'] = review['Body'].fillna(review['Subject'])
len(review[review['Body'].isnull()])

# Remove rows with both nan body and nan subject
review = review[~review['Body'].isnull()]

# Detect language

d = enchant.Dict("en_US")

def lang_detect(value):
    try:
        return d.check(value)
    except:
        return('error')

review['language_id'] = review['Body'].apply(lambda x: lang_detect(x))

len(review[~review['Translated Body'].isnull()])
len(review[~review['language_id']])

non_eng_review = review[~review['language_id']]

# CLD-2 is pretty good and extremely fast
# lang-detect is a tiny bit better, but much slower
# langid is good, but CLD-2 and lang-detect are much better
# NLTK's Textcat is neither efficient nor effective.

# combine English subject & body with translated subject & body
review['Subject_combined'] = review['Translated Subject'].fillna(review['Subject'])
review['Body_combined'] = review['Translated Body'].fillna(review['Body'])

review_set2 = review[['App Name', 'App Store', 'App', 'Store', 'App ID', 'Review ID',
       'Country', 'Version', 'Rating', 'Date', 'Author', 'Emotion', 'Device', 'Subject_combined', 'Body_combined']]

percent_missing = review_set2.isnull().sum()*100/len(review_set2)
missing_value_df = pd.DataFrame({'column_name': review_set2.columns,
                                 'percent_missing': percent_missing})