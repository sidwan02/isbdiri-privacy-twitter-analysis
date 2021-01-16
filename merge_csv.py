# https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/

import os
import glob
import pandas as pd
import csv
from datetime import datetime


def merge_all_csv():
    os.chdir(r'C:\Users\sidwa\OneDrive\OneDriveNew\Personal\Sid\Brown University\Internships\ISB\DIRI DS\privacy_data_sentiment_analysis\CSV')

    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv(
        r'C:\Users\sidwa\OneDrive\OneDriveNew\Personal\Sid\Brown University\Internships\ISB\DIRI DS\privacy_data_sentiment_analysis\results\consolidated.csv', index=False, encoding='utf-8-sig')

    with open(r"C:\Users\sidwa\OneDrive\OneDriveNew\Personal\Sid\Brown University\Internships\ISB\DIRI DS\privacy_data_sentiment_analysis\results\consolidated.csv", 'r', encoding="utf-8") as source:
        with open(r"C:\Users\sidwa\OneDrive\OneDriveNew\Personal\Sid\Brown University\Internships\ISB\DIRI DS\privacy_data_sentiment_analysis\results\consolidated_date_formatted.csv", 'w', encoding="utf-8") as result:
            writer = csv.writer(result, lineterminator='\n')
            reader = csv.reader(source)
            # source.readline() # this skips the header (not what we want)
            for row in reader:
                # print('row', row)
                date = row[1]
                try:
                    new_date = datetime.strptime(
                        date, '%m/%d/%Y').strftime("%Y-%m-%d")
                    # print('date', date)
                    if date != "":
                        row[1] = new_date
                except:  # the date is already correct format
                    pass
                writer.writerow(row)

    source.close()
    result.close()


# merge_all_csv()
