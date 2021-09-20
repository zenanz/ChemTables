## Table-BERT model

Here is the implementation of the Table-BERT model evaluated on the [_ChemTables_](https://doi.org/10.17632/g7tjh7tbrj.3) dataset. To run the model, create a folder named **data** to store the 5 folds of the dataset (named _ChemTables_fold_i_).

### Usage
Use of virtual environment (e.g. _virtualenv_, _Anaconda_) of **python 3.7** is recommended. To install dependencies, please run the following command in your python environment.
```bash
pip install -r requirements.txt
```
Pre-processing is needed before training, _preprocessing.py_ reads dataset from files and convert them to _TensorDataset_ s. Cache file for _TensorDataset_ s and label mapping dictionaries are stored in _cache/_. Run the following command to generate pre-processed datasets. Pre-processing mode can be selected from _linear_ and _natural_ and maximum input length can a integer from 1-512. See the paper for more information on these hyper-parameters.

```bash
python preprocessing.py [preprocessing_mode] [max_input_length]
```

Our code for fine-tuning tranformers does not support multi-GPU at the moment. Note that the code will automatically GPUs and run on **first GPUs available** in the current environment. To start training, please run

```bash
python train.py [preprocessing_mode] [max_input_length]
```

Pre-trained Table-BERT model on the ChemTables dataset can be downloaded from [here](https://chemu.eng.unimelb.edu.au/download/table-bert/).

To make inference with pre-trained model on a given dataset. Please put the folder of the pre-trained BERT model under **models** and run the following script. But you will need to make sure your dataset has been properly preprocessed and stored in **cache** folder.

```bash
python predict.py [model_folder_name] [preprocessing_mode]
```
