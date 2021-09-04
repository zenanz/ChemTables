import json
from sklearn.feature_extraction.text import TfidfVectorizer

def getStringAndLabelData(foldData, number_of_table_rows):

    labels, strings = [],[]
    for table in foldData:
        label = table['annotations']
        data = table['data']
        numrows = min(len(data), number_of_table_rows)

        tokens = []
        for row in data[:numrows]:
            for cell in row:
                if len(cell) > 0:
                    tokens += cell
        tablestring = ' '.join(tokens)

        labels.append(label)
        strings.append(tablestring)

    return strings,  labels

def readData(number_of_table_rows):
    fold1 = json.load(open("data/ChemTables_fold_1.json", 'r'))
    fold2 = json.load(open("data/ChemTables_fold_2.json", 'r'))
    fold3 = json.load(open("data/ChemTables_fold_3.json", 'r'))
    fold4 = json.load(open("data/ChemTables_fold_4.json", 'r'))
    fold5 = json.load(open("data/ChemTables_fold_5.json", 'r'))

    trainStrings, trainLabels = getStringAndLabelData(fold1 + fold2 + fold3, number_of_table_rows)
    devStrings, devLabels = getStringAndLabelData(fold4, number_of_table_rows)
    testStrings, testLabels = getStringAndLabelData(fold5, number_of_table_rows)

    return trainStrings, trainLabels, devStrings, devLabels, testStrings, testLabels

def extractFeatures(trainStrings, devStrings, testStrings, max_features):

    train_tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=max_features, min_df=2, norm='l2', token_pattern=r"(?u)\b[^ ]+\b")
    train_features = train_tfidf.fit_transform(trainStrings).toarray()

    dev_features = train_tfidf.transform(devStrings).toarray()
    test_features = train_tfidf.transform(testStrings).toarray()

    return train_features, dev_features, test_features

if __name__ == "__main__":
    trainStrings, trainLabels, devStrings, devLabels, testStrings, testLabels = readData(10)
    print(len(trainStrings), len(trainLabels), len(devStrings), len(devLabels), len(testStrings), len(testLabels))
