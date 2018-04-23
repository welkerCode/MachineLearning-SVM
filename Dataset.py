import csv
#import xlrd



from Example import Example
from Attributes import Attributes

class Dataset:

    def __init__(self,filename):
        self.filename = filename
        self.exampleList = []
        self.numAttributes = 0;
        self.attributeList = []
        self.parseCSV()

    '''
    def parseDatasheet(self):
        workbook = xlrd.open_workbook(self.filename)
        worksheet = workbook.sheet_by_index(0)
        num_rows = worksheet.nrows
        for i in range(0, num_rows):
            newExample = Example()
            newExample.parseAttributes(self.filename, i)
            self.numAttributes = len(newExample.getAttributes())
            self.exampleList.append(newExample)
        for attribute in self.exampleList[0].getAttributes():
            self.attributeList.append(Attributes())

    '''

    # This function handles CSV files rather than xls
    # I learned to parse CSV files from: /home/welkert/Documents/school/machine_learning/MachineLearning-DecisionTrees/Datasets_in_homework_1/test.csv/home/welkert/Documents/school/machine_learning/MachineLearning-DecisionTrees/Datasets_in_homework_1/train.csv
    def parseCSV(self):
        with open(self.filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                newExample = Example()
                newExample.parseAttributesFromCSV(row)
                self.numAttributes = len(newExample.getAttributes())
                self.exampleList.append(newExample)
            for attribute in self.exampleList[0].getAttributes():
                self.attributeList.append(Attributes())

    def createAttributeList(self):
        for example in self.exampleList:
            attributes = example.getAttributes()
            for x in range(0, len(attributes)):
                self.attributeList[x].addValue(attributes[x])

    def getAttributeList(self):
        return self.attributeList


    def getExampleList(self):
        return self.exampleList

    def getNumAttributes(self):
        return self.numAttributes