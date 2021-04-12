
from sklearn.model_selection import *
from sklearn.metrics import *
import numpy as np
import warnings, _pickle
warnings.filterwarnings('ignore')
random_state = 42
np.random.seed(random_state)

def run_with_validation(model,  train_features, trainLabels, dev_features, devLabels, test_features, testLabels,
                    problem_name: str,
                    logger=None) -> tuple:
    """
    Train classifier with k folds cross-validation setting
    :param model: the machine learning model (e.g., LinearSVM)
    :param problem_name: str
        The name of the classification problem (from the configuration file)
    :param n_w_features: int
        The number of word-based n_gram features to be extracted
    :param n_c_features: int
        The number of char-based n_gram features to be extracted
    :param n_lsi: int
        The size of the SVD model
    :param valid_size: float [0, 1], default 0.1
        The portion of data to be used to validate the model
    :param test_size: float [0, 1], default 0.1
        The portion of data to be used to test the model

    :param valid_size: float
        The portion of data to be used to validate the model
    :return: tuple
        The average Accuracy, Precision, Recoall and F1 scores (either macro or binary)
    """
    # features, labels = extract_features(problem_name, n_w_features, n_c_features, n_lsi, logger)
    #
    logger.info("Start training {} for {}".format(model, problem_name))
    #print("Start training {} for {}".format(model, problem_name))
    # n_classes = len(set(labels))
    # average = "binary" if n_classes == 2 else "macro"
    #
    # labels = np.asarray(labels)
    # x_train, y_train, x_test, y_test = _split_data(features, labels, test_size)
    # x_train, y_train, x_valid, y_valid = _split_data(x_train, y_train, valid_size)

    #train, dev, test = extractFeature_GreenButton("/Users/dqnguyen/Documents/patid_train.arff", "/Users/dqnguyen/Documents/patid_dev.arff", \
    #                "/Users/dqnguyen/Documents/patid_test.arff", 15000, 3000, logger)

    average = "macro"

    x_train, y_train = train_features, np.asarray(trainLabels)
    x_valid, y_valid = dev_features, np.asarray(devLabels)
    x_test, y_test = test_features, np.asarray(testLabels)

    model.fit(x_train, y_train)

    #modelName = "{}".format(model)
    #modelName = modelName[:modelName.find("(")]
    #pickle.dump([model, dev, test], open("/Users/dqnguyen/Documents/GreenButton/"+modelName, 'wb'))


    y_valid_pred = model.predict(x_valid)

    valid_accuracy, valid_f1, valid_precision, valid_recall = _eval_scores(average, y_valid, y_valid_pred)

    y_test_pred = model.predict(x_test)

    test_accuracy, test_f1, test_precision, test_recall = _eval_scores(average, y_test, y_test_pred)

    logger.info("Valid\tAccuracy: {}\tP: {}\tR: {}\tF1: {}\tReport: {}".format(valid_accuracy, valid_f1, valid_precision,
                                                                               valid_recall, classification_report(y_valid, y_valid_pred, digits=4)))
    logger.info("Test\tAccuracy: {}\tP: {}\tR: {}\tF1: {}\tReport: {}".format(test_accuracy, test_f1, test_precision,
                                                                              test_recall, classification_report(y_test, y_test_pred, digits=4)))
    return valid_accuracy, valid_f1, valid_precision, valid_recall


def _eval_scores(average, y_valid, y_valid_pred):
    P = precision_score(y_valid.tolist(), y_valid_pred.tolist(), average=average)
    R = recall_score(y_valid.tolist(), y_valid_pred.tolist(), average=average)
    F1 = f1_score(y_valid.tolist(), y_valid_pred.tolist(), average=average)
    Acc = accuracy_score(y_valid.tolist(), y_valid_pred.tolist())
    return Acc, F1, P, R


def _split_data(features, labels, size):
    data_splitter = StratifiedShuffleSplit(n_splits=1, random_state=random_state, test_size=size)
    for train_ids, test_ids in data_splitter.split(features, labels):

        x_train = features[train_ids]
        y_train = labels[train_ids]

        x_test = features[test_ids]
        y_test = labels[test_ids]

        return x_train, y_train, x_test, y_test
