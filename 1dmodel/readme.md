## Table-BERT model

Here is the implementation of the Table-BERT model evaluated on the _ChemTables_ dataset. To run the model, you should 

### Usage
Use of virtual environment (e.g. _virtualenv_, _Anaconda_) of **python 3.7** is recommended. To install dependencies, please run the following command in your python environment.
```bash
pip install -r requirements.txt
```
Pre-processing is needed before training, _preprocessing.py_ reads dataset from files and convert them to _TensorDataset_ s. Cache file for _TensorDataset_ s and label mapping dictionaries are stored in _cache/_. Run the following command to generate pre-processed datasets. Pre-processing mode can be selected from _linear_ and _natural_ and maximum input length can a integer from 1-512. See the paper for more information on these hyper-parameters.

```bash
python preprocessing.py [preprocessing_mode] [max_input_length]
```

Our code for fine-tuning tranformers supports multi-GPU training. Note that the code will automatically GPUs and run on **first GPUs available** in the current environment. To start training, please run

```bash
python train.py [preprocessing_mode] [max_input_length]
```

Pre-trained Table-BERT model on the ChemTables dataset can be downloaded from here.
