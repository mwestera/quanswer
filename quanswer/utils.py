
from transformers import pipeline
import functools


default_models = {
    'nl': 'raalst/RobBERT-v2-nl-ext-qa',
    'en': 'ahotrod/albert_xxlargev1_squad2_512',
}

@functools.cache
def load_qa_model(lang_or_name: str):
    model_name = default_models.get(lang_or_name) or lang_or_name
    qa_model = pipeline("question-answering", model=model_name)
    return qa_model