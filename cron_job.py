# https: // stackoverflow.com/questions/15088037/python-script-to-do-something-at-the-same-time-every-day
from datetime import datetime, timedelta
from threading import Timer
from privacy_history_multithreading import privacy_analysis_multithreading

x = datetime.today()
# chron job everyday 12 pm
y = x.replace(day=x.day, hour=12, minute=0, second=0,
              microsecond=0) + timedelta(days=1)
delta_t = y-x

secs = delta_t.total_seconds()


t = Timer(secs, privacy_analysis_multithreading)
t.start()
