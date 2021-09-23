import os
import re
import json
import copy
import argparse
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

def parse_table(table):

    def remove_spaces(cell):
        return re.sub('\s+', '', cell)


    rows = table.find_all('row')
    extracted_table = []
    meta_data = {}
    meta_data['tid'] = table['id']

    chemistry_list = []
    for i, r in enumerate(rows):
        extracted_row = []
        for j, entry in enumerate(r.find_all('entry')):
            if entry.find_all('chemistry') != 0:
                for chemistry in entry.find_all('chemistry'):
                    chemistry_list.append((i, j, str(chemistry)))

            if entry.has_attr('namest'):
#                 print(entry)
                try:
                    if entry['namest'].startswith('col'):
                        start = int(entry['namest'][3:])
                        end = int(entry['nameend'][3:])
                    elif entry['namest'] == 'offset':
                        start = len(extracted_row)
                        end = int(entry['nameend'])
                    else:
                        start = int(entry['namest'])
                        end = int(entry['nameend'])
                except:
                    print(table)

                span = end - start + 1
                extracted_row.append(remove_spaces(entry.get_text()))
                for i in range(1, span):
                    extracted_row.append('')
            else:
                extracted_row.append(remove_spaces(entry.get_text()))

        extracted_table.append(extracted_row)
    meta_data['chemistry'] = chemistry_list

    if len(table.find_all('title')) != 0:
        titles = [i.get_text() for i in table.find_all('title')]
        meta_data['titles'] = titles

    if len(table.find_all('thead')) != 0:
        header_list = []
        t = table.find_all('table')[0]
        childs = t.findChildren()
        child_names = [c.name for c in childs]

        row_count = 0
        in_thead = 0
        start = 0
        end = 0

        for i, c in enumerate(child_names):
            if c == 'row':
                row_count += 1
            # if start of thead element
            if c == 'thead':
                start = row_count
                in_thead = 1
            # if thead ends or table ends
            if (c == 'tbody' or i == len(child_names) - 1) and  in_thead == 1:
                end = row_count
                header_list.append((start, end))
                in_thead = 0
                start, end = 0, 0
        meta_data['thead'] = header_list

    return extracted_table, meta_data


def extract_all(root_dir):
    tables = []
    xml_list = os.listdir(root_dir)
    for pid in tqdm(xml_list):
        xml_path = os.path.join(root_dir, pid, pid+'.xml')
        if not os.path.exists(xml_path):
            continue
        ts = BeautifulSoup('\n'.join(open(xml_path, 'r').read().split('\n')[1:]), 'xml').find_all('tables')
        for t in ts:
            text, meta_data = parse_table(t)
            tables.append({
                'pid': pid,
                'tid': meta_data['tid'],
                'meta_data': meta_data,
                'data': text,
            })

    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract tables from XML-format patents.')
    parser.add_argument('xml_root', type=str,
                        help='root directory of the patent collection')
    parser.add_argument('output_path', type=str,
                        help='path of the output file')
    args = parser.parse_args()

    tables = extract_all(args.xml_root)
    json.dump(tables, open(os.path.join(args.output_path, 'extracted_tables.json'), 'w+'))
