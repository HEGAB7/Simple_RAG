import requests
from bs4 import BeautifulSoup as bs

URL = 'https://www.raisa.com/'

req = requests.get(URL)
soup = bs(req.text, 'html.parser')

text_spans = soup.find_all('span', class_ ="wixui-rich-text__text")
just_text = [text_span.get_text(strip=True) for text_span in text_spans]
just_text = [t for t in just_text if t != '\u200b']

text_str = ' '.join(just_text)

with open('Simple_RAG/Data/text.txt', 'w') as f:
  f.write(text_str)