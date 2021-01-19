# https://towardsdatascience.com/automate-your-python-scripts-with-task-scheduler-661d0a40b279
from privacy_history_multithreading import privacy_analysis_multithreading
from merge_csv import merge_all_csv
from TFIDF_counts import frequent_phrases

privacy_analysis_multithreading()
merge_all_csv()
frequent_phrases()
