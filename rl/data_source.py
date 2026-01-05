
import re
from datasets import load_dataset
from typing import Iterator, Optional
from abc import ABC, abstractmethod


def wikitext_detokenizer(text: str) -> str:
    """Detokenize WikiText exactly as lm-eval does."""
    string = text
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


class DataSource(ABC):
    """Abstract base class for data sources."""
    @abstractmethod
    def __iter__(self) -> Iterator[dict]:
        """Yield dictionaries with at least 'text' key."""
        ...
    @abstractmethod
    def __len__(self) -> int:
        ...


class WikiTextDataSource(DataSource):
    """
    WikiText data source for perplexity evaluation.
    
    Yields items with:
        - 'text': detokenized text (for model input/perplexity)
        - 'original_text': original text (for word counting)
        - 'word_count': number of words in original text
        - 'type': 'wikitext'
    """
    
    def __init__(self, split: str = "test", max_samples: Optional[int] = None):
        self.split = split
        self.max_samples = max_samples
        self._dataset = None
    
    def _load(self):
        if self._dataset is None:
            self._dataset = load_dataset(
                "EleutherAI/wikitext_document_level",
                "wikitext-2-raw-v1",
                split=self.split
            )
            if self.max_samples:
                self._dataset = self._dataset.select(range(min(self.max_samples, len(self._dataset))))
    
    def __iter__(self) -> Iterator[dict]:
        self._load()
        for item in self._dataset:
            original_text = item["page"]
            detokenized_text = wikitext_detokenizer(original_text)
            word_count = len(re.split(r"\s+", original_text))
            yield {
                "text": detokenized_text,
                "original_text": original_text,
                "word_count": word_count,
                "type": "wikitext",
            }
    
    def __len__(self) -> int:
        self._load()
        return len(self._dataset)


class MMLUDataSource(DataSource):
    """
    MMLU (Massive Multitask Language Understanding) data source.
    
    Yields items with:
        - 'text': formatted question with choices (for model input)
        - 'question': raw question text
        - 'choices': list of 4 answer choices
        - 'answer_idx': index of correct answer (0-3)
        - 'answer_letter': correct answer letter (A/B/C/D)
        - 'subject': MMLU subject/category
        - 'type': 'mmlu'
    """
    
    ANSWER_LETTERS = ["A", "B", "C", "D"]
    
    def __init__(
        self, 
        split: str = "test",
        subjects: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            split: 'test', 'validation', 'dev', or 'auxiliary_train'
            subjects: List of MMLU subjects to include. None = all subjects.
            max_samples: Maximum samples to load. None = all.
        """
        self.split = split
        self.subjects = subjects
        self.max_samples = max_samples
        self._dataset = None
    
    def _load(self):
        if self._dataset is None:            
            self._dataset = load_dataset("cais/mmlu", "all", split=self.split)            
            if self.subjects:
                self._dataset = self._dataset.filter(
                    lambda x: x["subject"] in self.subjects
                )
            if self.max_samples:
                self._dataset = self._dataset.select(
                    range(min(self.max_samples, len(self._dataset)))
                )
    
    def format_question(self, question: str, choices: list[str]) -> str:
        """Format question with lettered choices."""
        formatted = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            formatted += f"{self.ANSWER_LETTERS[i]}. {choice}\n"
        formatted += "\nAnswer:"
        return formatted
    
    def __iter__(self) -> Iterator[dict]:
        self._load()
        for item in self._dataset:
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]
            
            yield {
                "text": self.format_question(question, choices),
                "question": question,
                "choices": choices,
                "answer_idx": answer_idx,
                "answer_letter": self.ANSWER_LETTERS[answer_idx],
                "subject": item["subject"],
                "type": "mmlu",
            }
    
    def __len__(self) -> int:
        self._load()
        return len(self._dataset)
