import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob

# Step 1: Data Extraction

# Read input URLs from Input.xlsx 
input_data = pd.read_excel("C:\\Users\\admin\\Downloads\\Input.xlsx")
# the path mentioned is for my local machine 

# Function to extract text from a given URL
def extract_text(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find and extract the article text
        article_text = ""
        for paragraph in soup.find_all('p'):
            article_text += paragraph.get_text() + "\n"
        return article_text
    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return None

# Loop through each row in the input data
for index, row in input_data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    # Extract text from the URL
    text = extract_text(url)
    if text:
        # Save the extracted text into a text file
        with open(f"{url_id}.txt", "w", encoding="utf-8") as file:
            file.write(text)
            print(f"Text extracted from {url} and saved as {url_id}.txt")
    else:
        print(f"Text extraction failed for {url}")

# Step 2: Text Analysis

# Function to compute variables from text
def compute_variables(text):
    try:
        # Tokenize text into sentences and words
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Sentiment Analysis
        positive_score = len([word for word in words if word.lower() in positive_words])
        negative_score = len([word for word in words if word.lower() in negative_words])
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
        subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
        
        # Readability Analysis
        avg_sentence_length = len(words) / len(sentences)
        percentage_complex_words = len([word for word in words if len(word) > 6]) / len(words)
        fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
        
        # Additional Analysis
        avg_words_per_sentence = len(words) / len(sentences)
        complex_word_count = len([word for word in words if len(word) > 2])
        word_count = len(words)
        syllables_per_word = sum([len(re.findall('(?!e$)[aeiou]+', word.lower())) for word in words]) / word_count
        personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text))
        avg_word_length = sum(len(word) for word in words) / word_count
        
        return (positive_score, negative_score, polarity_score, subjectivity_score,
                avg_sentence_length, percentage_complex_words, fog_index,
                avg_words_per_sentence, complex_word_count, word_count,
                syllables_per_word, personal_pronouns, avg_word_length)
    except Exception as e:
        print(f"Error computing variables: {e}")
        return None

# Load stop words list
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Loading positive and negative words from the data provided. The files are locally stored in the machine
positive_words = set([word.strip() for word in open("C:\\Users\\admin\\Downloads\\PositiveWords.txt", "r") if word.strip() not in stop_words])
negative_words = set([word.strip() for word in open("C:\\Users\\admin\\Downloads\\NegativeWords.txt", "r") if word.strip() not in stop_words])

# Loop through each text file and compute variables
output_data = []
for index, row in input_data.iterrows():
    url_id = row['URL_ID']
    try:
        # Read the text from the file
        with open(f"{url_id}.txt", "r", encoding="utf-8") as file:
            text = file.read()
            # Compute variables
            variables = compute_variables(text)
            if variables:
                output_data.append([url_id] + list(variables))
                print(f"Variables computed for {url_id}.txt")
            else:
                print(f"Variable computation failed for {url_id}.txt")
    except Exception as e:
        print(f"Error processing {url_id}.txt: {e}")

# Converting output data to DataFrame
output_df = pd.DataFrame(output_data, columns=['URL_ID', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
                                                'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
                                                'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
                                                'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'])

# Saving output to excel
output_df.to_excel("output.xlsx", index=False)
print("Output saved to output.xlsx")

