## COMP90051 Project 1 Tweet Authorship Attribution

We leverages transformer-based language models pre-trained on massive corpus to extract semantic features of tweets, and perform authorship attribution by fine-tuning the pre-trained LM on classification tasks.

### Usage
Use of virtual environment (e.g. _virtualenv_, _Anaconda_) of **python 3.7** is recommended. To install dependencies, please run the following command in your python environment.
```bash
pip install -r requirements.txt
```
Pre-processing is needed before training, _preprocessing.py_ reads dataset from files and convert them to _TensorDataset_ s. Cache file for _TensorDataset_ s and label mapping dictionaries are stored in _cache/_. Run the following command to generate pre-processed datasets.

```bash
python preprocessing.py
```
Our code for fine-tuning tranformers supports multi-GPU training. Note that the code will automatically detect number of GPUs and run on **all GPUs available** in the current environment. Please specify visible GPUs in your environment variables to avoid holding up all GPUs. To start training, please run

```bash
python train.py
```

The predicted labels of tweets in test set are saved in _test_preds.txt_. To generate _.csv_ file ready for Kaggle submission, please run

```bash
python create_csv.py
```
