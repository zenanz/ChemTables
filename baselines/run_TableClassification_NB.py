from sklearn.svm import *
from xgboost.sklearn import *
from sklearn.multiclass import *
from sklearn.linear_model import *
from sklearn.neural_network import *
from xgboost.sklearn import *
from sklearn.ensemble import *
from sklearn.naive_bayes import *
import argparse, os
from readData import *
from training_classifier import *
import logging, datetime

def create_folder_if_not_exist(folder_path: str) -> None:
    """
    Create a folder if not exist
    :param folder_path: str
    :return:
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except:
            pass
    return None

def create_logger(logger_name: str = "Table_classification",
                  file_name: str = None,
                  level: int = logging.INFO) -> logging.RootLogger:
    logging_folder = "{}/log_files/{}".format(os.path.dirname(os.path.abspath(__file__)), logger_name)
    create_folder_if_not_exist(logging_folder)

    logging_file = "{}/{}.log".format(logging_folder, datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S.%f")
                                      if file_name is None else file_name)
    logger = logging.getLogger(logger_name)
    hdlr = logging.FileHandler(logging_file)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    # formatter = logging.Formatter("%(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(level)
    return logger

if __name__ == "__main__":

    #model = OneVsRestClassifier(XGBClassifier(n_estimators=200, n_jobs=8, objective="binary:logistic"))
    #model = XGBClassifier(silent=False)
    #model = MLPClassifier()
    #model = XGBRegressor()

    #model = SVC(C=0.9, kernel="linear", max_iter=500, verbose=0, random_state=0)


    problem_name = "Table_classification"
    logger = create_logger(logger_name=problem_name)

    #if eval_mode == "kfolds":
    #    run_with_kfolds(model, n_w_features=n_w_features, n_c_features=n_c_features, n_lsi=n_lsi, problem_name=problem_name, logger=logger)
    #else:



    #
    # train, dev, test = extractFeature_GreenButton("/Users/dqnguyen/Documents/GreenButton/patid_train_150.arff",
    #                                               "/Users/dqnguyen/Documents/GreenButton/patid_dev_150.arff", \
    #                                               "/Users/dqnguyen/Documents/GreenButton/patid_test_150.arff",
    #                                               10000, 1000, logger)
    #
    # #{'C': 0.3, 'loss': 'squared_hinge', 'random_state': 0, 'penalty': 'l2', 'max_iter': 200, 'class_weight': 'balanced'}
    #
    # model = LinearSVC(C=0.3, max_iter=200, verbose=0, random_state=0, penalty='l2', loss="squared_hinge",
    #                    class_weight="balanced")
    # run_with_validation(model, train, dev, test, problem_name=problem_name,  n_w_features=n_w_features, n_c_features=n_c_features, n_lsi=n_lsi, logger=logger)



    number_of_table_rows = 10
    max_features = 1000


    run = 0
    for number_of_table_rows in [1, 2, 3]:#[5, 10, 15, 20, 25, 50, 100]:

        trainStrings, trainLabels, devStrings, devLabels, testStrings, testLabels = readData(number_of_table_rows)

        for max_features in [9000, 10000, 11000, 12000, 13000, 14000, 15000]: #[500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000]:

            train_features, dev_features, test_features = extractFeatures(trainStrings, devStrings, testStrings, max_features)

            logger.info("problem_name: {}, number_of_table_rows: {}, max_features: {}"
                        .format(problem_name, number_of_table_rows, max_features))

            for model in [ComplementNB()]:#[ComplementNB(), GaussianNB(), MultinomialNB(),  BernoulliNB()]:

                run += 1
                print(run)

                run_with_validation(model, train_features, trainLabels, dev_features, devLabels, test_features, testLabels, problem_name=problem_name, logger=logger)


    # for n_w_features in [10000, 12000, 14000, 15000, 16000, 18000, 20000]:
    #     for n_c_features in [1000, 1500, 2000, 2500, 3000]:
    #         for numhis in ['150', "200", "250", "300"]:
    #             train, dev, test = extractFeature_GreenButton("/Users/dqnguyen/Documents/arffFiles/patid_train_" + numhis + ".arff",
    #                                                   "/Users/dqnguyen/Documents/arffFiles/patid_dev_" + numhis + ".arff", \
    #                                                   "/Users/dqnguyen/Documents/arffFiles/patid_test_" + numhis + ".arff", n_w_features,
    #                                                       n_c_features, logger)
    #             model = BernoulliNB()
    #             run_with_validation(model, train, dev, test, problem_name=problem_name, n_w_features=n_w_features,
    #                     n_c_features=n_c_features, n_lsi=n_lsi, logger=logger)
    # optimal: 10000, 1000, 150
    '''
    features, labels = load_saved_data("/Users/dqnguyen/Dropbox/workspace/Others/data/saved/primary_tumour/1000_1000_200.pkl")

    count = 0
    for feature, label in zip(features, labels):
        print(label, feature)
        count += 1
        if count > 10:
            break
            

    '''
