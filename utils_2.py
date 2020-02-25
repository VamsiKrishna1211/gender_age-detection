import datetime as date
from dateutil.relativedelta import relativedelta

# This method compares two dates and return the difference in years
def compare_date(start, end):
    try:
        d1 = date.datetime.strptime(start[0:10], '%Y-%m-%d')
        d2 = date.datetime.strptime(str(end), '%Y')
        rdelta = relativedelta(d2, d1)
        diff = rdelta.years
    except Exception as ex:
        #print(ex)
        diff = -1

    return diff

# Cleans the full date
def clean_full_date(dt):
    temp = dt.split('-')
    year = temp[0]
    month = temp[1]
    day = temp[2]

    if (len(year) != 4):
        year = '0000'

    if (len(month) == 1):
        month = '0' + month

    if (month == '00'):
        month = '01'

    if (len(day) == 1):
        day = '0' + day

    if (day == '00'):
        day = '01'

    return year + '-' + month + '-' + day
