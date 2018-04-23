# I learned to parse workbooks with https://www.sitepoint.com/using-python-parse-spreadsheet-data/


class Attributes:

    def __init__(self):
        self.values = []

    def addValue(self, newValue):
        if self.values.count(newValue) == 0:
            self.values.append(newValue)

    def getValues(self):
        return self.values