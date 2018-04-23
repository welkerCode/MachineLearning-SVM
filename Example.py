# I learned to parse workbooks with https://www.sitepoint.com/using-python-parse-spreadsheet-data/


#import xlrd

class Example:

    def __init__(self):
        self.attributes = []
        self.label = None

    def addAttribute(self, newAttribute):
        self.attributes.append(newAttribute)

    '''
    def parseAttributes(self, workbook_file, row_index):
        workbook = xlrd.open_workbook(workbook_file, row_index)
        worksheet = workbook.sheet_by_index(0)
        for i in range(0,worksheet.row_len(row_index)-1):
            self.attributes.append(worksheet.cell(row_index, i).value)
        self.label = worksheet.cell(row_index, worksheet.row_len(row_index)-1).value
    '''

    def parseAttributesFromCSV(self, row):
        for i in range(0, len(row)-1):
            self.attributes.append(row[i])
        self.label = row[len(row)-1]

    # Getters and Setters
    def getLabel(self):
        return self.label

    def getAttributes(self):
        return self.attributes

    def getAttributeValue(self, index):
        return self.attributes[index]

    def getNumAttributes(self):
        return len(self.attributes)