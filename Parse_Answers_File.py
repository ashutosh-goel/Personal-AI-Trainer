import os
from typing import List
import re

def parse_answers_file(file_path: str) -> List[str]:
    """
    Parses a text file for student answers.
    Answers are expected to be separated by two or more newlines.
    A block of text containing only whitespace is treated as a skipped question.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Answer file not found at {file_path}. All questions will be marked as unanswered.")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split answers by 2 or more newlines
    # This treats each paragraph block as a potential answer
    potential_answers = re.split(r'\n\s*\n', content)
    
    # Clean up answers: strip whitespace and return
    return [answer.strip() for answer in potential_answers]