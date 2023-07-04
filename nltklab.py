import urllib.request


url = "https://www.gutenberg.org/files/2701/2701-0.txt"
file = urllib.request.urlopen(url)

text = [line.decode('utf-8') for line in file]
text = ''.join(text)

print(text[7600:8000])


import nltk 
nltk.download('punkt')
