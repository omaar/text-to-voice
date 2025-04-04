import re

class SpellCheckerManager:
    def __init__(self, text=None):
        self.text = text

    def fix_double_spaces(self):
        # Reemplazar múltiples espacios por un solo espacio
        self.text = re.sub(r'\s{2,}', ' ', self.text)

    def fix_spaces_around_punctuation(self):
        # Arreglar espacios antes o después de guiones largos y signos
        self.text = re.sub(r'\s*—\s*', ' — ', self.text)
        self.text = re.sub(r'\s*([,:;.])', r'\1', self.text)

    def correct_text(self, text=""):
        if text:
            self.text = text

        self.fix_double_spaces()
        self.fix_spaces_around_punctuation()
        return self.text