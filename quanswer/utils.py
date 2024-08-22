from transformers.pipelines import QuestionAnsweringPipeline, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import pipeline
import functools
import numpy as np


default_models = {
    'nl': 'raalst/RobBERT-v2-nl-ext-qa',
    'en': 'ahotrod/albert_xxlargev1_squad2_512',
}

@functools.cache
def load_qa_model(lang_or_name: str, return_token_scores=True):
    model_name = default_models.get(lang_or_name) or lang_or_name
    if return_token_scores:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_model = QuestionAnsweringPipelineTokenProbs(model=model, tokenizer=tokenizer)
    else:
        qa_model = pipeline("question-answering", model=model_name)
    return qa_model



class QuestionAnsweringPipelineTokenProbs(QuestionAnsweringPipeline):
    def postprocess(
        self,
        model_outputs,
        *args,
        align_to_words=True,
        handle_impossible_answer=False,
        **kwargs,
    ):
        results = super().postprocess(model_outputs, *args, align_to_words=align_to_words, handle_impossible_answer=handle_impossible_answer, **kwargs)
        if not isinstance(results, list):
            results = [results]

        for result, output in zip(results, model_outputs):

            probas, spans, prob_unanswered = self.get_per_token_probas(output, align_to_words, handle_impossible_answer)

            result['token_scores'] = probas
            result['token_spans'] = spans

            if handle_impossible_answer:
                result['is_answered'] = 1.0 - prob_unanswered

        return results


    def get_per_token_probas(self, output, align_to_words, handle_impossible_answer):

        start_proba = softmax(output['start'].squeeze()).tolist()
        end_proba = softmax(output['end'].squeeze()).tolist()

        example = output['example']

        token_start_probas = {}
        token_end_probas = {}
        token_spans = {}

        # Following copied from super: turning word pieces back into genuine words (taking its max proba)
        if not self.tokenizer.is_fast:
            char_to_word = np.array(example.char_to_word_offset)
            token_to_orig_map = output["token_to_orig_map"]
            for token_i, (start_logit, end_logit) in enumerate(zip(start_proba, end_proba)):
                token_i_orig = token_to_orig_map[token_i]
                token_start_probas.setdefault(token_i_orig, []).append(start_logit)
                token_end_probas.setdefault(token_i_orig, []).append(end_logit)
                token_spans[token_i_orig] = (np.where(char_to_word == token_i_orig)[0][0].item(),
                                             np.where(char_to_word == token_i_orig)[0][-1].item())

        else:
            question_first = bool(self.tokenizer.padding_side == "right")
            enc = output["encoding"]
            if self.tokenizer.padding_side == "left":
                offset = (output["input_ids"] == self.tokenizer.pad_token_id).numpy().sum()
            else:
                offset = 0
            sequence_index = 1 if question_first else 0

            for token_i, (start_logit, end_logit) in enumerate(zip(start_proba, end_proba)):
                token_i_orig = enc.token_to_word(token_i - offset)
                token_start_probas.setdefault(token_i_orig, []).append(start_logit)
                token_end_probas.setdefault(token_i_orig, []).append(end_logit)
                token_spans[token_i_orig] = self.get_indices(enc, token_i, token_i, sequence_index, align_to_words)

        token_start_probas = {key: max(value) for key, value in token_start_probas.items()}
        token_end_probas = {key: max(value) for key, value in token_end_probas.items()}

        if handle_impossible_answer:
            prob_unanswered = max(token_start_probas[None], token_end_probas[None])
            del token_start_probas[None]
            del token_end_probas[None]
            del token_spans[None]
        else:
            prob_unanswered = None

        # Compute per-token probability as:
        #   for each token, loop through all possible starting points (before it), and compute the probability of
        #   a span starting there that includes the token (i.e., ending somewhere after it).
        #   then sum all those probs for each token.
        start_probs = list(token_start_probas.values())
        end_probs = list(token_end_probas.values())
        token_probs = []
        for token_i_orig in range(len(start_probs)):
            probs = []
            for start_i_orig in range(0, token_i_orig + 1):
                sum_possible_end_probs = sum(end_probs[start_i_orig:])    # TODO: Potentially optimizable by using tensors?
                sum_inclusive_possible_end_probs = sum(end_probs[token_i_orig:])
                prob_inclusive = start_probs[start_i_orig] * (sum_inclusive_possible_end_probs/sum_possible_end_probs)
                probs.append(prob_inclusive)
            token_probs.append(sum(probs))

        return token_probs, list(token_spans.values()), prob_unanswered


def softmax(x):
    exps = np.exp(x)
    return exps / sum(exps)