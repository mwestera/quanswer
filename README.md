# QuAnswer: CLI for question answering pipelines #

A command-line wrapper around [transformers](https://huggingface.co/docs/transformers/), to apply a question answering model.

For English, default is [ahotrod/albert_xxlargev1_squad2_512](https://huggingface.co/ahotrod/albert_xxlargev1_squad2_512). Good on SquadV2 and seemed decent on Reddit data.

For Dutch, default is [RobBERT-v2-nl-ext-qa](https://huggingface.co/raalst/RobBERT-v2-nl-ext-qa), seems okay on translated Squadv2 but much worse than English... There's a new version of RobBERT (2023), but not yet finetuned on QA.

## Install ##

Recommended is to first install pipx for ease of installing Python command line tools:

`pip install pipx`

Then: 

`pipx install git+https://github.com/mwestera/quanswer`

This will make the command `quanswer` available in your shell.

## Usage ##

Input should be .csv with columns context question, or .jsonl with 'context' and 'question' fields, e.g.: 

```bash
quanswer some_qa_items.jsonl > score.csv
```

Or feed a single context,question csv pair:

```bash
$ echo "\"The capital of France is Paris, while the capital of Holland is Amsterdam, and so on.\",What is the capital of NL?" | quanswer > score.csv
```

To get full json output, include `--json`: 

```bash
quanswer some_more_qa_items.csv --json > results.jsonl
```

