from datetime import date
from calendar import monthrange
f1 = date(1957, 1, 31)
f2 = date(1958, 12, 31)
def addmonth(fecha):
    year = fecha.year
    month = fecha.month
    if month < 12:
        month += 1
    else:
        year += 1
        month = 1
    fecha1 = date(year, month, 1)
    day = monthrange(year, month)[1]
    return date(year, month, day)
while f1<=f2:
    print(f'{f1}')
    f1 = addmonth(f1)
print('fin')
