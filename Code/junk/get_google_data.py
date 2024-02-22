# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:52:53 2016

@author: mike
"""

# Collect google hit results for each drug with and without amnesia
from numpy import mod
from utils.data_funcs import read_drug_data
import pandas
from googleapiclient.discovery import build
from tqdm import tqdm
import time

dar, dnar, _ = read_drug_data()
drug_list = dar.index.tolist()
raw_rec = pandas.DataFrame(0, columns=drug_list, index=["amnesia", "total"])

# Build a service object for interacting with the API. Visit
# the Google APIs Console <http://code.google.com/apis/console>
# to get an API key for your own application.
service = build("customsearch", "v1",
            developerKey=DEV_KEY_SECRET)

def amnesia_search_hits(drug_name):
    drug_res = service.cse().list(
          q=drug_name + " drug",
          cx = '008969237201105559499:y_nac8lru8u',
        ).execute()
    drug_hit = int(drug_res['searchInformation']['totalResults'])
    amnesia_res = service.cse().list(
          q=drug_name + " anmesia",
          cx = '008969237201105559499:y_nac8lru8u',
        ).execute()
    amnesia_hit = int(amnesia_res['searchInformation']['totalResults'])
    return pandas.Series({"amnesia":amnesia_hit, "total":drug_hit})

#~ main_rec = pandas.read_pickle("{STORAGE_DIR}/Data/googback/goog_data_drug_checkpoint_800.pickle")
#~ raw_rec = main_rec.T
#~ for i in tqdm(range(195, len(drug_list))):

for i in tqdm(range(len(drug_list))):
for i in tqdm(range(1820, len(drug_list))):
    drug_name = drug_list[i]
    raw_rec[drug_name] = amnesia_search_hits(drug_name)
    if mod(i, 200) ==0:
        tqdm.write(drug_name +" " + str(i) + "\n" + str(raw_rec[drug_name]))
        main_rec = raw_rec.T
        main_rec["ratio"] = main_rec["amnesia"].astype(float) / main_rec["total"]
        main_rec.to_pickle("{STORAGE_DIR}/Data/googback/goog_data_drug_checkpoint_{0}.pickle".format(i))
    time.sleep(0.05)

main_rec = raw_rec.T
main_rec["ratio"] = main_rec["amnesia"].astype(float) / main_rec["total"]
main_rec
main_rec.to_pickle("{STORAGE_DIR}/Data/googback/GoogleSearchHitWDrugData.pickle")
