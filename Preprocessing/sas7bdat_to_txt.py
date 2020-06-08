import sys
from sas7bdat import SAS7BDAT


with SAS7BDAT(sys.argv[1]) as f:
    for row in f:
        myString = "|".join(str(v) for v in row)
        print(myString)
