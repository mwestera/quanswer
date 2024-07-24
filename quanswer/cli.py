#!/usr/bin/python
import csv

import itertools
import sys
import argparse
import json
from typing import Iterable, Callable, List

from transformers.data.processors import SquadExample

from .utils import *

import io


"""
CLI wrapper around Huggingface transformers' pipeline for question answering.

Example:

$ echo "\"The capital of France is Paris, while the capital of Holland is Amsterdam, and so on.\",What is the capital of NL?" | quanswer --json --lang nl --topk 3

"""


def main():

    parser = argparse.ArgumentParser(description='Wrapper around question answering transformers pipeline.')
    parser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='File containing json or csv lines with keys "context" and "question", or "context,question" csv pairs.')
    parser.add_argument('--model', '--lang', type=str, default='en', help='Language code or specific model to use; default en.')
    parser.add_argument('--topk', type=int, default=1, help='How many answer candidats to return per item.')
    parser.add_argument('--json', action='store_true', help='Whether to output full results as json. Otherwise outputs only the score.')
    parser.add_argument('--mustanswer', action='store_true', help='To disallow non-answers (span 0,0), like for Squad v1.')
    parser.add_argument('--tokenscores', action='store_true', help='Whether to return per-token scores as well.')

    args = parser.parse_args()

    qa_model = load_qa_model(args.model, return_logits=args.tokenscores)

    for result in qa_model(reader(args.file), handle_impossible_answer=not args.mustanswer, top_k=args.topk):

        if args.topk > 1:
            answers = result
            result = {}
            if not args.mustanswer:
                result['is_answered'] = answers[0]['is_answered']
                del answers[0]['is_answered']
            result.update({
                'score': answers[0]['score'],
                'start': answers[0]['start'],
                'end': answers[0]['end'],
                'answer': answers[0]['answer'],
                'answers': answers,
            })
            if args.tokenscores:
                result['token_scores'] = answers[0]['token_scores']
                result['token_spans'] = answers[0]['token_spans']
                del answers[0]['token_scores']
                del answers[0]['token_spans']

        if args.json:
            # TODO: Make keys ordering consistent with the above case
            print(json.dumps(result))   # with keys: score, start, stop, answer, (token_scores, token_spans, is_answered)
        else:
            print(result['score'])  # TODO: consider returning only is_answered


def reader(file):
    file, is_json = peek_if_jsonl(file)
    if is_json:
        basereader = (json.loads(line.strip()) for line in file)
    else:  # assume csv
        file, header = strip_csv_header(file, header_contains=['context', 'question'])
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
    firstline = next(file)
    file = StringIteratorIO(itertools.chain([firstline], file)) # TODO: Works, but if using a custom class, better just make a peekable version of stdin
    try:
        d = json.loads(firstline.strip())
        if isinstance(d, dict):
            return file, True
    except json.JSONDecodeError:
        pass
    return file, False


def strip_csv_header(file, header_contains):
    firstline = next(file)
    parsed = next(csv.reader([firstline.strip()]))
    if all(c in parsed for c in header_contains):
        return file, parsed
    else:  # means it was no header
        return StringIteratorIO(itertools.chain([firstline], file)), None


class StringIteratorIO(io.TextIOBase):
    """
    https://stackoverflow.com/questions/12593576/adapt-an-iterator-to-behave-like-a-file-like-object-in-python
    """

    def __init__(self, iter):
        self._iter = iter
        self._left = ''

    def readable(self):
        return True

    def _read1(self, n=None):
        while not self._left:
            try:
                self._left = next(self._iter)
            except StopIteration:
                break
        ret = self._left[:n]
        self._left = self._left[len(ret):]
        return ret

    def read(self, n=None):
        l = []
        if n is None or n < 0:
            while True:
                m = self._read1()
                if not m:
                    break
                l.append(m)
        else:
            while n > 0:
                m = self._read1(n)
                if not m:
                    break
                n -= len(m)
                l.append(m)
        return ''.join(l)

    def readline(self):
        l = []
        while True:
            i = self._left.find('\n')
            if i == -1:
                l.append(self._left)
                try:
                    self._left = next(self._iter)
                except StopIteration:
                    self._left = ''
                    break
            else:
                l.append(self._left[:i+1])
                self._left = self._left[i+1:]
                break
        return ''.join(l)


if __name__ == '__main__':
    main()
