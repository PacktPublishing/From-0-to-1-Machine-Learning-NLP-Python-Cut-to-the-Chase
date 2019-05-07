
# coding: utf-8

# In[ ]:

# Let's import the nltk module
import nltk

# To start, we need some text to play with. NLTK has many corpora and resources for you to explore natural language. 
# A one-off run of nltk.download() will get you all the resources in one go. Once you've done that you should have 
# a repository of interesting texts including stuff like Moby Dick and an Inaugural Address Corpus

from nltk.book import *


# In[ ]:

# These texts have now been loaded and you can refer to them by their names. These are objects of type 'Text' and they have a
# bunch of cool methods to explore the text 

# concordance will print all the occurrences of a word along with some context. Let's explore two texts - Moby Dick and 
# Sense and Sensibility. As expected, word usage and language in both these books are pretty different :) 

text1.concordance("monstrous")


# In[ ]:

text2.concordance("monstrous")


# In[ ]:

# As you can see, Melville uses the word 'monstrous' in a different connotation than Austen. He uses it to indicate
# size and things that are terrifying, Austen uses it in a positive connotation
# Let's see what other words appear in the same context as monstrous
text2.similar("monstrous")


# In[ ]:

# Clearly Austen uses "monstrous" to represent positive emotions and to amplify those emotions. She seems to use it 
# interchangeably with "very"  
text2.common_contexts(["monstrous","very"])


# In[ ]:

# These are fun ways to explore the usage of natural language in different contexts or situations. Let's see how the 
# usage of certain words by Presidents has changed over the years. 
# (Do install matplotlib before you run the below line of code)
text4.dispersion_plot(["citizens","democracy","freedom","duties","America"])


# In[ ]:

# Let's see what kind of emotions are expressed in Jane Austen's works vs Herman Melville's
text2.dispersion_plot(["happy","sad"])


# In[ ]:

text1.dispersion_plot(["happy","sad"])


# In[ ]:

# Now let's get to some serious stuff. Often you want to extract features from 
# a text - these are attributes that will represent the text - words or sentences 
# How do we split a piece of text into constituent sentences/words? (these are called tokens)
from nltk.tokenize import word_tokenize, sent_tokenize
text="Mary had a little lamb. Her fleece was white as snow"
sents=sent_tokenize(text)
print(sents)


# In[ ]:

words=[word_tokenize(sent) for sent in sents]
print(words)


# In[ ]:

# Let's filter out stopwords (words that are very common like 'was', 'a', 'as etc)
from nltk.corpus import stopwords 
from string import punctuation
customStopWords=set(stopwords.words('english')+list(punctuation))
#Notice how we made the stopwords a set

wordsWOStopwords=[word for word in word_tokenize(text) if word not in customStopWords]
print(wordsWOStopwords)


# In[ ]:

text2="Mary closed on closing night when she was in the mood to close."
# 'close' appears in different morphological forms here, stemming will reduce all forms of the word 'close' to its root
# NLTK has multiple stemmers based on different rules/algorithms. Stemming is also known as lemmatization. 
from nltk.stem.lancaster import LancasterStemmer
st=LancasterStemmer()
stemmedWords=[st.stem(word) for word in word_tokenize(text2)]
print(stemmedWords)


# In[ ]:

# NLTK has functionality to automatically tag words as nouns, verbs, conjunctions etc
nltk.pos_tag(word_tokenize(text2))


# In[ ]:




# In[ ]:




# In[ ]:



