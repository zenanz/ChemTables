## 2 Dimensional Models for ChemTables

Here is the implementation of _TabNet_ and _TBResNet_ evaluated on the [_ChemTables_](https://doi.org/10.17632/g7tjh7tbrj.3) dataset. To run the model, create a folder named **data** to store the 5 folds of the dataset (named _rxTables_tokenized_fold_i_). Then, download [ChemPatent pre-trained word embeddings](https://chemu.eng.unimelb.edu.au/patent_w2v/) and include under this folder (2dmodels/).

### Usage
Use of virtual environment (e.g. _virtualenv_, _Anaconda_) of **python 3.7** is recommended. To install dependencies, please run the following command in your python environment.
```bash
pip install -r requirements.txt
```

To train the model, run the following command, the argument model can be selected from _resnet_ or _tabnet_. The maximum row and columns are not limited as long as the input size could fit in your GPU memory.

```bash
python train.py [model] [max_rows] [max_columns]
```

Trained TabNet and TBResNet model state dicts on the ChemTables dataset can be downloaded from [here](https://chemu.eng.unimelb.edu.au/download/table-bert/).

To make inference with the trained model, run the following script

```bash
python predict.py [model] [max_rows] [max_columns] [path_to_model_state_dict]
```
