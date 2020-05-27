# -*- coding: utf-8 -*-

import pandas as pd

def data_preprocess(raw_data):
    
    data = []
    
    for lines in raw_data:
        line = lines.split('\t')
        
        review = text_preprocess(line[0].lower())
        sentiment = int(line[1][0])
        
        row = []
        row.append(review)
        row.append(sentiment)
        data.append(row)
    
    return pd.DataFrame(data, columns=['Review', 'Sentiment'])

def text_preprocess(text):
    
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    # text = stem(text)
    
    return text

# Remove punctuations
def remove_punctuations(text):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    
    for char in text:
        if char in punctuations:
            text = text.replace(char, "")
    
    return text

#Remove Stop words
def remove_stopwords(text):
    
    # removing words such as 'would' 'does' 'very' 'have' because they can be associated with both positive and negative words
    
    stopwords = ['im', 'ourselves', 'hers', 'yourself', 'there', 'during', 'they', 'own', 'an', 'be',
                 'for', 'its', 'yours', 'of', 'itself', 'is', 's', 'am', 'who', 'as', 'from', 'him', 'the',
                 'themselves', 'are', 'we', 'these', 'your', 'his', 'me', 'were', 'her', 'himself', 'this',
                 'our', 'their', 'above', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'them', 'and',
                 'in', 'yourselves', 'then', 'that', 'he', 'you', 'herself', 'myself', 'those', 'i', 'ive',
                 'whom', 't', 'being', 'theirs', 'my', 'a', 'by', 'it', 
                  'phone', 'product', ' ', '', 'headset', 'battery', 'ear', 'ears', 'cell', 'camera', 'people', 
                  'bluetooth', 'device', 'would', 'does', 'sound', '2', 'very', 'so', 'use','have', 'one', 
                  'case', 'motorola', 'service', 'or', 'buy']
    
    text_words = text.split(' ')
    filtered_words = [w for w in text_words if w not in stopwords]
    
    # print(filtered_words)
    filtered_text = ' '.join(filtered_words)
    
    return filtered_text

    
def stem(text):
    
    text_words = text.split(' ');
    for i in range(len(text_words)):
        word = text_words[i]
        if len(word) == 0:
            continue
        if word[-1] == 's':
            text_words[i] = word[:-1]
            
    text = ' '.join(text_words)
    
    return text



