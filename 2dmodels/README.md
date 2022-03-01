## 2 Dimensional Models for ChemTables

Here is the implementation of _TabNet_ and _TBResNet_ evaluated on the [_ChemTables_](https://doi.org/10.17632/g7tjh7tbrj.3) dataset. To run the model, create a folder named **data** to store the 5 folds of the dataset (named _rxTables_tokenized_fold_i_). Then, download [ChemPatent pre-trained word embeddings](https://chemu.eng.unimelb.edu.au/patent_w2v/) and include under this folder (2dmodels/).

### Usage
Use of virtual environment (e.g. _virtualenv_, _Anaconda_) of **python 3.7** is recommended. To install dependencies, please run the following command in your python environment.
```bash
pip install -r requirements.txt
```

To train the model, run the following command, the argument model can be selected from _resnet_ or _tabnet_. The maximum row and columns are not limited as long as the input size could fit in your GPU memory.

Mode **full** stands for standard 5 fold dataset split in which the first 3 folds are for training while 4th and 5th fold are for validation and test respectively/

Mode **no_dev** stands for the baseline setup for ChEMU shared task, this mode uses first 4 folds as training set and the last 1 for evaluation.

Mode **inference** stands for prediction mode. Please make sure you have [_ChemTables_] dataset in **data** folder and data for inference in a new folder named **test_data**. Trained TabNet and TBResNet model state dicts on the ChemTables dataset can be downloaded from [here](https://chemu.eng.unimelb.edu.au/download/table-bert/).

```bash
python train.py [model] [max_rows] [max_columns] --mode [mode name] --weight_path [weight path]
```
