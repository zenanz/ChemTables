## Baseline Models for ChemTables

Here is the implementation of _SVM_ and _Naive Bayes_ evaluated on the [_ChemTables_](https://data.mendeley.com/datasets/g7tjh7tbrj) dataset. To run the model, create a folder named **data** to store the 5 folds of the dataset (named _ChemTables_fold_i_).

### Usage
Use of virtual environment (e.g. _virtualenv_, _Anaconda_) of **python 3.7** is recommended. To install dependencies, please run the following command in your python environment.
```bash
pip install -r requirements.txt
```

To train the _SVM_ model, run the following command

```bash
python run_TableClassification_LinearSVC.py
```

To train the _Naive Bayes_ model, run the following command


```bash
run_TableClassification_NB.py
```

Results can then be found under **log_files/**.
