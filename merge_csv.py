# https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/

import os
import glob
import pandas as pd


def merge_all_csv():
    os.chdir(r'C:\Users\sidwa\OneDrive\OneDriveNew\Personal\Sid\Brown University\Internships\ISB\DIRI DS\privacy_data_sentiment_analysis\CSV')

    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    combined_csv.to_csv(
        r'C:\Users\sidwa\OneDrive\OneDriveNew\Personal\Sid\Brown University\Internships\ISB\DIRI DS\privacy_data_sentiment_analysis\results\consolidated.csv', index=False, encoding='utf-8-sig')


merge_all_csv()
