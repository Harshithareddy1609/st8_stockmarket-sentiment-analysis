import re
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
nltk.download("punkt_tab")
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
class texttonum:
    def __init__(self,text):
        self.text=text
    def cleaner(self):
        cleaned_text = re.sub(r'[^\w\s]', '',self.text)  # Removes everything except word characters and spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replaces multiple spaces with a single space
        cleaned_data=cleaned_text.strip()  # Removes leading/trailing whitespace
        self.cleaned=cleaned_data
    def token(self):
        self.tkns=word_tokenize(self.cleaned)
    def removestop(self):
        stop=stopwords.words('english')
        self.cl=[i for i in self.tkns if i not in stop]
    def stem(self):
        ps=PorterStemmer()
        self.st=[ps.stem(word) for word in self.cl]
        return self.st