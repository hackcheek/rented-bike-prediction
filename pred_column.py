"""
This script extract 'pred' column from a csv file. 
Then write other csv with this column

Usage
-----
>>> pred_column.sh input.csv output.csv
"""
import sys
import csv

from typing import Iterable


errmsg = """
ERROR: I need a csv file as input and output
Usage: pred_column.sh input.csv output.csv
"""


def from_csv(csvfile: str):
    with open(csvfile, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield int(row['pred'])


def to_csv(content: Iterable, output_file: str):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pred'])
        for row in content:
            writer.writerow([row])


if __name__ == '__main__':

    try:
        input_file = sys.argv[1]
        output_file = sys.argv[2]

    except IndexError as err:
        print(errmsg)
        sys.exit(1)

    content = from_csv(input_file) 
    to_csv(content, output_file)
