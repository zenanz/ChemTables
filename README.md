# ChemTables
Models evaluated in "ChemTables: A dataset for semantic classificationon tables in chemical patents"

## 1 dimensional model
Implementation of Table-BERT

## 2 dimensional models
Implementation of TabNet and TBResNet

## Baselines
Implementation of SVM and Naive Bayes baselines

## Extract tables from XML-format patents
Name your downloaded patents in XML format with their patent IDs. Then wrap each individual patent with a folder named by its patent ID. Then run the following script

```bash
python extract_tables_from_xmls.py [xml_root] [output_path]
```
