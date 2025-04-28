import os
import re
import textwrap
from functools import cached_property

import torch
from tokenizers import Tokenizer

# Hindi/Devanagari digit mapping
_hindi_digits = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# Hindi ordinal regex
_hindi_ordinal_re = re.compile(r"([0-9०-९]+)(वा|वी|वे)")

# Update the Hindi-specific regex patterns
_hindi_number_re = re.compile(r'[0-9०-९]+')
_hindi_decimal_number_re = re.compile(r'([0-9०-९]+[.,][0-9०-९]+)')
_hindi_comma_number_re = re.compile(r'\b[0-9०-९]{1,3}(,[0-9०-९]{2,3})+(\.[0-9०-९]+)?\b')
_hindi_currency_re = re.compile(r'((₹[0-9०-९\.\,]*[0-9०-९]+)|([0-9०-९\.\,]*[0-9०-९]+₹))')

# Hindi number words for different values
_hindi_number_words = {
    0: 'शून्य', 1: 'एक', 2: 'दोन', 3: 'तीन', 4: 'चार', 5: 'पाच',
    6: 'सहा', 7: 'सात', 8: 'आठ', 9: 'नऊ', 10: 'दहा',
    11: 'अकरा', 12: 'बारा', 13: 'तेरा', 14: 'चौदा', 15: 'पंधरा',
    16: 'सोळा', 17: 'सतरा', 18: 'अठरा', 19: 'एकोणीस', 20: 'वीस',
    21: 'एकवीस', 22: 'बावीस', 23: 'तेवीस', 24: 'चोवीस', 25: 'पंचवीस',
    26: 'सव्वीस', 27: 'सत्तावीस', 28: 'अठ्ठावीस', 29: 'एकोणतीस', 30: 'तीस',
    31: 'एकतीस', 32: 'बत्तीस', 33: 'तेहतीस', 34: 'चौतीस', 35: 'पस्तीस',
    36: 'छत्तीस', 37: 'सदतीस', 38: 'अडतीस', 39: 'एकोणचाळीस', 40: 'चाळीस',
    41: 'एक्केचाळीस', 42: 'बेचाळीस', 43: 'त्रेचाळीस', 44: 'चव्वेचाळीस', 45: 'पंचेचाळीस',
    46: 'शेहेचाळीस', 47: 'सत्तेचाळीस', 48: 'अठ्ठेचाळीस', 49: 'एकोणपन्नास', 50: 'पन्नास',
    51: 'एक्कावन्न', 52: 'बावन्न', 53: 'त्रेपन्न', 54: 'चौपन्न', 55: 'पंचावन्न',
    56: 'छप्पन्न', 57: 'सत्तावन्न', 58: 'अठ्ठावन्न', 59: 'एकोणसाठ', 60: 'साठ',
    61: 'एकसष्ट', 62: 'बासष्ट', 63: 'त्रेसष्ट', 64: 'चौसष्ट', 65: 'पासष्ट',
    66: 'सहासष्ट', 67: 'सदुसष्ट', 68: 'अडुसष्ट', 69: 'एकोणसत्तर', 70: 'सत्तर',
    71: 'एक्काहत्तर', 72: 'बाहत्तर', 73: 'त्र्याहत्तर', 74: 'चौर्‍याहत्तर', 75: 'पंच्याहत्तर',
    76: 'शहात्तर', 77: 'सत्याहत्तर', 78: 'अठ्याहत्तर', 79: 'एकोणऐंशी', 80: 'ऐंशी',
    81: 'एक्क्याऐंशी', 82: 'ब्याऐंशी', 83: 'त्र्याऐंशी', 84: 'चौऱ्याऐंशी', 85: 'पंच्याऐंशी',
    86: 'शहाऐंशी', 87: 'सत्त्याऐंशी', 88: 'अठ्ठ्याऐंशी', 89: 'एकोणनव्वद', 90: 'नव्वद',
    91: 'एक्क्याण्णव', 92: 'ब्याण्णव', 93: 'त्र्याण्णव', 94: 'चौऱ्याण्णव', 95: 'पंच्याण्णव',
    96: 'शहाण्णव', 97: 'सत्त्याण्णव', 98: 'अठ्ठ्याण्णव', 99: 'नव्व्याण्णव', 100: 'शंभर'
}

# For original code compatibility - keeping multilingual dictionaries but with Hindi focus
_ordinal_re = {
    "hi": re.compile(r"([0-9०-९]+)(वां|वीं|वे|था|थी|थे)"),
    # Adding minimal stubs for other languages to prevent key errors
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
}

# Hindi symbols mapping
_symbols_multilingual = {
    "hi": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " आणि "),
            ("@", " अॅट "),
            ("%", " टक्के "),
            ("#", " हॅश "),
            ("$", " डॉलर "),
            ("£", " पाउंड "),
            ("€", " युरो "),
            ("₹", " रुपये "),
            ("°", " अंश "),
        ]
    ],
    # Placeholders for other languages to prevent key errors
    "en": [],
    "es": [],
    "fr": [],
    "de": [],
    "pt": [],
    "it": [],
    "pl": [],
    "ar": [],
    "cs": [],
    "ru": [],
    "nl": [],
    "tr": [],
    "hu": [],
    "ko": [],
    "zh": []
}

# Hindi abbreviations
_abbreviations = {
    "hi": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("डॉ", "डॉक्टर"),
            ("श्री", "श्रीमान"),
            ("श्रीमती", "श्रीमती"),
            ("प्रा", "प्राध्यापक"),
            ("सौ", "सौभाग्यवती"),
            ("कु", "कुमारी"),
        ]
    ],
    # Placeholders for other languages to prevent key errors
    "en": [],
    "es": [],
    "fr": [],
    "de": [],
    "pt": [],
    "it": [],
    "pl": [],
    "ar": [],
    "cs": [],
    "ru": [],
    "nl": [],
    "tr": [],
    "hu": [],
    "ko": [],
    "zh": []
}

_whitespace_re = re.compile(r"\s+")

_bank_account_re = re.compile(r'\b(?:खाता संख्या|अकाउंट नंबर|अकाउंट नम्बर|खाता नंबर|खाता नम्बर|a/c|account number|account no)[:\s]+([0-9०-९\s]{9,18})\b', re.IGNORECASE)

# Function to handle bank account numbers
def _expand_bank_account(m):
    """Read bank account numbers digit by digit"""
    account_num = normalize_hindi_digits(m.group(1).replace(" ", ""))
    result = ""
    
    # Read each digit separately with small pauses
    for i, digit in enumerate(account_num):
        digit_num = int(digit)
        result += _hindi_number_words[digit_num]
        
        # Group digits in pairs for better pronunciation
        if i < len(account_num) - 1 and i % 2 == 1:
            result += " "
    
    return result

def normalize_hindi_digits(text):
    """Convert Hindi/Devanagari digits to Arabic numerals"""
    for hindi_digit, arabic_digit in _hindi_digits.items():
        text = text.replace(hindi_digit, arabic_digit)
    return text

def hindi_number_to_words(num):
    """Convert a number to Marathi words using Indian numbering system"""
    if num < 0:
        return "ऋण " + hindi_number_to_words(abs(num))
    if num == 0:
        return "शून्य"
    
    # For small numbers, use direct mapping
    if num <= 100 and num in _hindi_number_words:
        return _hindi_number_words[num]
    
    # Handle numbers according to Marathi number system
    if num < 100:
        # For numbers not in the dictionary (should never happen with our complete dict)
        return str(num)
    elif num < 1000:
        hundreds = num // 100
        remainder = num % 100
        if remainder:
            return _hindi_number_words[hundreds] + "शे " + hindi_number_to_words(remainder)
        else:
            return _hindi_number_words[hundreds] + "शे"
    elif num < 100000:
        thousands = num // 1000
        remainder = num % 1000
        if thousands == 1:
            prefix = "एक हजार"
        else:
            prefix = hindi_number_to_words(thousands) + " हजार"
        if remainder:
            return prefix + " " + hindi_number_to_words(remainder)
        else:
            return prefix
    elif num < 10000000:
        lakhs = num // 100000
        remainder = num % 100000
        if lakhs == 1:
            prefix = "एक लाख"
        else:
            prefix = hindi_number_to_words(lakhs) + " लाख"
        if remainder:
            return prefix + " " + hindi_number_to_words(remainder)
        else:
            return prefix
    else:
        crores = num // 10000000
        remainder = num % 10000000
        if crores == 1:
            prefix = "एक कोटी"
        else:
            prefix = hindi_number_to_words(crores) + " कोटी"
        if remainder:
            return prefix + " " + hindi_number_to_words(remainder)
        else:
            return prefix

def _expand_decimal_point(m, lang="hi"):
    """Handle decimal numbers in Marathi"""
    text = m.group(1)
    text = normalize_hindi_digits(text)
    
    if ',' in text:
        text = text.replace(',', '.')
    
    integer_part, decimal_part = text.split('.')
    integer_num = int(integer_part)
    
    result = hindi_number_to_words(integer_num)
    
    # For decimal part, read each digit individually
    if decimal_part:
        result += " पूर्णांक "
        for digit in decimal_part:
            digit_num = int(digit)
            result += _hindi_number_words[digit_num] + " "
    
    return result.strip()

def _expand_hindi_ordinal(m):
    """Handle ordinal numbers in Hindi"""
    num = normalize_hindi_digits(m.group(1))
    suffix = m.group(2)
    
    number_word = hindi_number_to_words(int(num))
    
    # Apply appropriate ordinal suffix based on gender
    if suffix in ['वां', 'था']:
        return number_word + "वां"
    elif suffix in ['वीं', 'थी']:
        return number_word + "वीं"
    else:
        return number_word + "वें"

def _expand_hindi_currency(m):
    """Handle currency expressions in Marathi"""
    amount_text = m.group(0).replace('₹', '').replace(',', '')
    amount_text = normalize_hindi_digits(amount_text)
    
    # Handle decimal point
    if '.' in amount_text:
        rupees, paise = amount_text.split('.')
        rupees_int = int(rupees)
        paise_int = int(paise) if len(paise) <= 2 else int(paise[:2])
        
        rupees_text = hindi_number_to_words(rupees_int)
        
        if paise_int > 0:
            paise_text = hindi_number_to_words(paise_int)
            # Use direct connection without "और" to avoid pauses
            return f"{rupees_text} रुपये {paise_text} पैसे"
        else:
            return f"{rupees_text} रुपये"
    else:
        amount = int(amount_text)
        amount_text = hindi_number_to_words(amount)
        return f"{amount_text} रुपये"

def _expand_hindi_number(m):
    """Handle general Hindi numbers"""
    text = m.group(0).replace(',', '')
    text = normalize_hindi_digits(text)
    number = int(text)
    return hindi_number_to_words(number)

def expand_symbols_multilingual(text, lang="hi"):
    """Expand symbols into Hindi words"""
    if lang == "hi":
        for regex, replacement in _symbols_multilingual[lang]:
            text = re.sub(regex, replacement, text)
            text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()

def expand_abbreviations_multilingual(text, lang="hi"):
    """Expand Hindi abbreviations"""
    if lang == "hi":
        for regex, replacement in _abbreviations[lang]:
            text = re.sub(regex, replacement, text)
    return text

def expand_numbers_multilingual(text, lang="hi"):
    """Handle Hindi numbers comprehensively"""
    if lang == "hi":
        # Handle currency with Hindi digits
        text = re.sub(_hindi_currency_re, _expand_hindi_currency, text)
        
        # Handle decimal numbers with Hindi digits
        text = re.sub(_hindi_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
        
        # Handle ordinal numbers with Hindi digits
        text = re.sub(_ordinal_re[lang], _expand_hindi_ordinal, text)
        
        # Handle regular numbers with Hindi digits
        text = re.sub(_hindi_number_re, _expand_hindi_number, text)
    
    return text

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)

def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def hindi_cleaners(text, lang="hi"):
    """Hindi-specific text cleaning pipeline with improved handling"""
    text = text.replace('"', "")
    text = text.lower()
    
    # Check for bank account numbers first
    text = re.sub(_bank_account_re, _expand_bank_account, text)
    
    # Then handle regular numbers
    text = expand_numbers_multilingual(text, lang)
    text = expand_abbreviations_multilingual(text, lang)
    text = expand_symbols_multilingual(text, lang)
    
    # Handle recurring digits like phone numbers
    def identify_digit_sequences(match):
        digits = normalize_hindi_digits(match.group(0))
        # If it looks like a phone number (10+ digits) or a sequence that
        # shouldn't be converted to words, read digit by digit
        if len(digits) >= 10 or re.match(r'\d+[-/]\d+', digits):
            result = ""
            for i, digit in enumerate(digits):
                if digit.isdigit():
                    digit_num = int(digit)
                    result += _hindi_number_words[digit_num] + " "
                else:
                    result += digit + " "
            return result.strip()
        return match.group(0)  # Let the regular number expansion handle it
    
    # Find potential digit sequences (phone numbers, codes, etc.)
    digit_seq_pattern = re.compile(r'\b[0-9०-९-/]{10,}\b')
    text = re.sub(digit_seq_pattern, identify_digit_sequences, text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Keep the original split_sentence function signature but implement for Hindi
def split_sentence(text, lang="hi", text_split_length=250):
    """Split Hindi text into sentences with improved pausing"""
    text_splits = []
    
    # Add explicit pause markers for common punctuation
    text = text.replace(",", ", ")  # Add space after commas for slight pause
    text = text.replace(":", ": ")  # Add space after colons
    
    # Better sentence splitting with multiple delimiters
    sentences = re.split(r'([।\n\?\!])', text)
    
    # Rejoin sentences with their ending punctuation
    i = 0
    rejoined_sentences = []
    while i < len(sentences):
        if i+1 < len(sentences) and sentences[i+1] in ['।', '?', '!']:
            rejoined_sentences.append(sentences[i] + sentences[i+1])
            i += 2
        else:
            rejoined_sentences.append(sentences[i])
            i += 1
    
    current_split = ""
    
    for sentence in rejoined_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_split) + len(sentence) <= text_split_length:
            current_split += sentence + " "
        elif len(sentence) > text_split_length:
            # For long sentences, use more natural breaking points
            if current_split:
                text_splits.append(current_split.strip())
                current_split = ""
                
            # Break at more natural points like commas, conjunctions
            
            natural_breaks = re.split(r'([,;]|\sआणि\s|\sपण\s|\sकी\s|\sतर\s)', sentence)
            sub_current = ""
            
            i = 0
            while i < len(natural_breaks):
                part = natural_breaks[i]
                next_break = natural_breaks[i+1] if i+1 < len(natural_breaks) else ""
                
                if len(sub_current) + len(part) + len(next_break) <= text_split_length:
                    sub_current += part + next_break
                    i += 2 if next_break else 1
                else:
                    if sub_current:
                        text_splits.append(sub_current.strip())
                        sub_current = ""
                    
                    # If a single part is too long, use wrap
                    if len(part) > text_split_length:
                        for line in textwrap.wrap(
                            part,
                            width=text_split_length,
                            drop_whitespace=True,
                            break_on_hyphens=False,
                        ):
                            text_splits.append(line)
                    else:
                        sub_current = part
                        i += 1
            
            if sub_current:
                text_splits.append(sub_current.strip())
        else:
            if current_split:
                text_splits.append(current_split.strip())
            current_split = sentence + " "
    
    if current_split:
        text_splits.append(current_split.strip())

    return text_splits

# Keep the original VoiceBpeTokenizer class structure
DEFAULT_VOCAB_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/tokenizer.json")

class VoiceBpeTokenizer:
    def __init__(self, vocab_file=None):
        self.tokenizer = None
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)
        self.char_limits = {
            "en": 250,
            "hi": 500,
        }

    @cached_property
    def katsu(self):
        import cutlet
        return cutlet.Cutlet()

    def check_input_length(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        limit = self.char_limits.get(lang, 250)
        if len(txt) > limit:
            print(
                f"[!] Warning: The text length exceeds the character limit of {limit} for language '{lang}', this might cause truncated audio."
            )

    def preprocess_text(self, txt, lang):
        if lang == "hi":
            txt = hindi_cleaners(txt, lang)
        else:
            # For non-Hindi languages, just do basic cleaning
            txt = basic_cleaners(txt)
        return txt

    def encode(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        self.check_input_length(txt, lang)
        txt = self.preprocess_text(txt, lang)
        lang = "zh-cn" if lang == "zh" else lang
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1

def test_hindi_processing():
    """Test function for Hindi text processing"""
    test_cases = [
        # Basic Marathi digits
        ("माझ्याकडे ५० रुपये आहेत.", "माझ्याकडे पन्नास रुपये आहेत."),
        # Mixed Marathi-English digits
        ("माझ्याकडे 50 रुपये आहेत.", "माझ्याकडे पन्नास रुपये आहेत."),
        # Currency with Marathi digits
        ("₹१५०", "एक शे पन्नास रुपये"),
        # Currency with commas and decimals
        ("₹१,५०,०००.५०", "एक लाख पन्नास हजार रुपये पन्नास पैसे"),
        # Decimal numbers
        ("३.१४", "तीन पूर्णांक एक चार"),
        # Ordinals
        ("५वा", "पाचवा"),
        # Number with commas (Indian style)
        ("१,२५,००,०००", "एक कोटी पंचवीस लाख"),
        # Symbols
        ("माझ्याकडे 14% बॅटरी आहे.", "माझ्याकडे चौदा टक्के बॅटरी आहे."),
    ]
    
    for input_text, expected_output in test_cases:
        cleaned_text = hindi_cleaners(input_text)
        print(f"Input: {input_text}")
        print(f"Expected: {expected_output}")
        print(f"Actual: {cleaned_text}")
        print(f"Test {'passed' if cleaned_text == expected_output else 'failed'}")
        print("-" * 50)

if __name__ == "__main__":
    test_hindi_processing()

