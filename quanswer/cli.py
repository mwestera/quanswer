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


FLOAT_PRECISION = 5

def main():

    class RoundingFloat(float):
        __repr__ = staticmethod(lambda x: format(x, f'.{FLOAT_PRECISION}f'))
    json.encoder.c_make_encoder = None
    json.encoder.float = RoundingFloat

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
        else:
            result = result[0]

        if args.json:
            keys_order = ['is_answered', 'score', 'start', 'end', 'answer', 'answers', 'token_scores', 'token_spans']
            result = {key: result[key] for key in keys_order if key in result}
            print(json.dumps(result))
        else:
            if args.mustanswer:
                print(result['score'])
            else:
                print(result['is_answered'])

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
    file, peekfile = itertools.tee(file)
    firstline = next(peekfile)
    try:
        d = json.loads(firstline.strip())
        if isinstance(d, dict):
            return file, True
    except json.JSONDecodeError:
        pass
    return file, False


def strip_csv_header(file, header_contains):
    file, peekfile = itertools.tee(file)
    firstline = next(peekfile)
    parsed = next(csv.reader([firstline.strip()]))
    if all(c in parsed for c in header_contains):
        return peekfile, parsed
    else:  # means it was no header
        return file, None


if __name__ == '__main__':
    main()
