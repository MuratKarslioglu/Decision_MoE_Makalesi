"""Kalite kuralları (Dataset Pipeline §5, kural 1 ve 3 + prompt-örnek sızıntısı + T4 olumsuzluk).

Hem üretim sırasında (zayıf çıktıyı yeniden üretmek için) hem filtre aşamasında kullanılır.
Prompt değiştirilmez; bu modül yalnızca üretilen çıktıyı değerlendirir.
"""
import re
from difflib import SequenceMatcher

MIN_WORDS = 5
MAX_WORDS = 40

# Kural 3: bu tiplerde bu cevaplar yasak (open-ended olmalı)
FORBIDDEN_ANSWERS_FOR = {
    "T1": {"yes", "no", "none", "true", "false"},
    "T2": {"yes", "no", "none"},
    "T3": {"yes", "no", "none"},
}

# Prompttaki örnek sorular: modelin bunları görüntüye bakmadan kopyalaması tespit edilir.
# T1 örneği ("How many people ...") ve T4 örneği doğal sorular olduğundan kontrol dışı.
PROMPT_EXAMPLES = {
    "T2": ["Is the red car to the left or right of the bridge?"],
    "T3": ["What color is the object on the table, and what does that color typically represent?"],
    "T5": ["Does the sign show a price above 10 dollars?"],
}
LEAK_THRESHOLD = 0.85

NEGATION_STARTS = ("no", "none", "not", "nothing", "there is no", "there are no", "never")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def word_count_ok(question: str) -> bool:
    return MIN_WORDS <= len(question.split()) <= MAX_WORDS


def answer_allowed(qtype: str, answer: str) -> bool:
    return normalize(answer) not in FORBIDDEN_ANSWERS_FOR.get(qtype, set())


def leaks_prompt_example(qtype: str, question: str) -> bool:
    q = normalize(question)
    return any(SequenceMatcher(None, q, normalize(e)).ratio() >= LEAK_THRESHOLD for e in PROMPT_EXAMPLES.get(qtype, []))


def t4_is_negation(answer: str) -> bool:
    a = normalize(answer)
    return any(a == s or a.startswith(s + " ") for s in NEGATION_STARTS)


def check(qtype: str, question: str, answer: str) -> tuple[bool, str]:
    """(geçerli mi, ret nedeni). Neden geçerliyse boş string."""
    if not word_count_ok(question):
        return False, "length"
    if not answer_allowed(qtype, answer):
        return False, "forbidden_answer"
    if leaks_prompt_example(qtype, question):
        return False, "prompt_example_leak"
    if qtype == "T4" and not t4_is_negation(answer):
        return False, "t4_not_negation"
    return True, ""
