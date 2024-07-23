#!/usr/bin/python
import csv

import itertools
import sys
import argparse
import json
from typing import Iterable, Callable, List

from transformers.data.processors import SquadExample

from .utils import *

"""
CLI wrapper around Huggingface transformers' pipeline for question answering.

Example:

$ echo "What is the capital of NL?,\"The capital of France is Paris, while the capital of Holland is Amsterdam, and so on.\"" | quanswer --json > results.jsonl
"""


def main():

    parser = argparse.ArgumentParser(description='Wrapper around question answering transformers pipeline.')
    parser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help='File containing json lines with keys "context" and "question", or "context,question" csv pairs.')
    parser.add_argument('--model', '--lang', type=str, default='en', help='Language code or specific model to use; default en.')    # TODO: Add langdetect?
    parser.add_argument('--json', action='store_true', help='Whether to output full results as json. Otherwise outputs only the score.')

    args = parser.parse_args()

    qa_model = load_qa_model(args.model)

    for items in batched(reader(args.file), 10000):
        results = qa_model(items, handle_impossible_answer=True)
        if args.json:
            print(json.dumps(results))
        else:
            print(results['score'])


def reader(file):
    file, is_json = peek_if_jsonl(file)
    if is_json:
        basereader = (json.loads(line.strip()) for line in file)
    else:  # assume csv
        file, header = strip_csv_header(file)
        basereader = csv.DictReader(file, fieldnames=header if header else ['context', 'question'])

    for n, item in enumerate(basereader):
        yield SquadExample(
            qas_id=item.get('id', n),
            question_text=item['question'],
            context_text=item['context'],
            answer_text=item.get('answer_text'),
            start_position_character=item.get('start_position_character'),
            title=item.get('title'),
        )


def peek_if_jsonl(file):
    firstline = next(file).strip()
    file = itertools.chain([firstline], file)
    try:
        d = json.loads(firstline)
        if isinstance(d, dict):
            return file, True
    except json.JSONDecodeError:
        pass
    return file, False


def strip_csv_header(file):
    firstline = next(file).strip()
    parsed = next(csv.reader([firstline]))
    if 'context' in parsed and 'question' in parsed:
        return file, parsed
    else:  # means it was no header
        return itertools.chain([firstline], file), None


# Included here, since only available in Python 3.12...
def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


if __name__ == '__main__':
    main()
