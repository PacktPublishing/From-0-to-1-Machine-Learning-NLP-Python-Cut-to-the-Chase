
# coding: utf-8

# In[ ]:


######################################################################################
# THis example is pretty much entirely based on this excellent blog post
# http://glowingpython.blogspot.in/2014/09/text-summarization-with-nltk.html
# Thanks to TheGlowingPython, the good soul that wrote this excellent article!
# That blog is is really interesting btw.
######################################################################################


######################################################################################
# nltk - "natural language toolkit" is a python library with support for 
#         natural language processing. Super-handy.
# Specifically, we will use 2 functions from nltk
#  sent_tokenize: given a group of text, tokenize (split) it into sentences
#  word_tokenize: given a group of text, tokenize (split) it into words
#  stopwords.words('english') to find and ignored very common words ('I', 'the',...) 
#  to use stopwords, you need to have run nltk.download() first - one-off setup
######################################################################################
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords

######################################################################################
# We have use dictionaries so far, but now that we have covered classes - this is a good
# time to introduce defaultdict. THis is class that inherits from dictionary, but has
# one additional nice feature: Usually, a Python dictionary throws a KeyError if you try 
# to get an item with a key that is not currently in the dictionary. 
# The defaultdict in contrast will simply create any items that you try to access 
# (provided of course they do not exist yet). To create such a "default" item, it relies 
# a function that is passed in..more below. 
######################################################################################
from collections import defaultdict

######################################################################################
#  punctuation to ignore punctuation symbols
######################################################################################
from string import punctuation

######################################################################################
# heapq.nlargest is a function that given a list, easily and quickly returns
# the 'n' largest elements in the list. More below
######################################################################################
from heapq import nlargest


######################################################################################
# Our first class, named FrequencySummarizer 
######################################################################################
class FrequencySummarizer:
    # indentation changes - we are now inside the class definition
    def __init__(self, min_cut=0.1, max_cut=0.9):
        # The constructor named __init__
        # THis function will be called each time an object of this class is 
        # instantiated
        # btw, note how the special keyword 'self' is passed in as the first
        # argument to each method (member function).
        self._min_cut = min_cut
        self._max_cut = max_cut 
        # Words that have a frequency term lower than min_cut 
        # or higer than max_cut will be ignored.
        self._stopwords = set(stopwords.words('english') + list(punctuation))
        # Punctuation symbols and stopwords (common words like 'an','the' etc) are ignored
        #
        # Here self._min_cut, self._max_cut and self._stopwords are all member variables
        # i.e. each object (instance) of this class will have an independent version of these
        # variables. 
        # Note how this function is used to set up the member variables to their appropriate values
    # indentation changes - we are out of the constructor (member function, but we are still inside)
    # the class.
    # One important note: if you are used to programming in Java or C#: if you define a variable here
    # i.e. outside a member function but inside the class - it becomes a STATIC member variable
    # THis is an important difference from Java, C# (where all member variables would be defined here)
    # and is a common gotcha to be avoided.

    def _compute_frequencies(self, word_sent):
        # next method (member function) which takes in self (the special keyword for this same object)
        # as well as a list of sentences, and outputs a dictionary, where the keys are words, and
        # values are the frequencies of those words in the set of sentences
        freq = defaultdict(int)
        # defaultdict, which we referred to above - is a class that inherits from dictionary,
        # with one difference: Usually, a Python dictionary throws a KeyError if you try 
        # to get an item with a key that is not currently in the dictionary. 
        # The defaultdict in contrast will simply create any items that you try to access 
        # (provided of course they do not exist yet). THe 'int' passed in as argument tells
        # the defaultdict object to create a default value of 0
        for s in word_sent:
        # indentation changes - we are inside the for loop, for each sentence
          for word in s:
            # indentation changes again - this is an inner for loop, once per each word_sent
            # in that sentence
            if word not in self._stopwords:
                # if the word is in the member variable (dictionary) self._stopwords, then ignore it,
                # else increment the frequency. Had the dictionary freq been a regular dictionary (not a 
                # defaultdict, we would have had to first check whether this word is in the dict
                freq[word] += 1
        # Done with the frequency calculation - now go through our frequency list and do 2 things
        #   normalize the frequencies by dividing each by the highest frequency (this allows us to 
        #            always have frequencies between 0 and 1, which makes comparing them easy
        #   filter out frequencies that are too high or too low. A trick that yields better results.
        m = float(max(freq.values()))
        # get the highest frequency of any word in the list of words
        
        for w in list(freq.keys()):
            # indentation changes - we are inside the for loop
            freq[w] = freq[w]/m
            # divide each frequency by that max value, so it is now between 0 and 1
            if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
                # indentation changes - we are inside the if statement - if we are here the word is either
                # really common or really uncommon. In either case - delete it from our dictionary
                del freq[w]
                # remember that del can be used to remove a key-value pair from the dictionary
        return freq
        # return the frequency list

    def summarize(self, text, n):
        # next method (member function) which takes in self (the special keyword for this same object)
        # as well as the raw text, and the number of sentences we wish the summary to contain. Return the 
        # summary
        sents = sent_tokenize(text)
        # split the text into sentences
        assert n <= len(sents)
        # assert is a way of making sure a condition holds true, else an exception is thrown. Used to do 
        # sanity checks like making sure the summary is shorter than the original article.
        word_sent = [word_tokenize(s.lower()) for s in sents]
        # This 1 sentence does a lot: it converts each sentence to lower-case, then 
        # splits each sentence into words, then takes all of those lists (1 per sentence)
        # and mushes them into 1 big list
        self._freq = self._compute_frequencies(word_sent)
        # make a call to the method (member function) _compute_frequencies, and places that in
        # the member variable _freq. 
        ranking = defaultdict(int)
        # create an empty dictionary (of the superior defaultdict variety) to hold the rankings of the 
            # sentences. 
        for i,sent in enumerate(word_sent):
            # Indentation changes - we are inside the for loop. Oh! and this is a different type of for loop
            # A new built-in function, enumerate(), will make certain loops a bit clearer. enumerate(sequence), 
            # will return (0, thing[0]), (1, thing[1]), (2, thing[2]), and so forth.
            # A common idiom to change every element of a list looks like this:
            #  for i in range(len(L)):
            #    item = L[i]
            #    ... compute some result based on item ...
            #    L[i] = result
            # This can be rewritten using enumerate() as:
            # for i, item in enumerate(L):
            #    ... compute some result based on item ...
            #    L[i] = result
            for w in sent:
                # for each word in this sentence
                if w in self._freq:
                    # if this is not a stopword (common word), add the frequency of that word 
                    # to the weightage assigned to that sentence 
                    ranking[i] += self._freq[w]
        # OK - we are outside the for loop and now have rankings for all the sentences
        sents_idx = nlargest(n, ranking, key=ranking.get)
        # we want to return the first n sentences with highest ranking, use the nlargest function to do so
        # this function needs to know how to get the list of values to rank, so give it a function - simply the 
        # get method of the dictionary
        return [sents[j] for j in sents_idx]
       # return a list with these values in a list
# Indentation changes - we are done with our FrequencySummarizer class!


######################################################################################
# Now to get a URL and summarize
######################################################################################
import urllib.request
from bs4 import BeautifulSoup

######################################################################################
# Introducing Beautiful Soup: " Beautiful Soup is a Python library for pulling data out of 
# HTML and XML files. It works with your favorite parser to provide idiomatic ways of 
# navigating, searching, and modifying the parse tree. It commonly saves programmers hours 
# or days of work.
######################################################################################



def get_only_text_washington_post_url(url):
    # This function takes in a URL as an argument, and returns only the text of the article in that URL. 
    page = urllib.request.urlopen(url).read().decode('utf8')
    # download the URL
    soup = BeautifulSoup(page)
    # initialise a BeautifulSoup object with the text of that URL
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    # use this code to get everything in that text that lies between a pair of 
    # <article> and </article> tags. We do this because we know that the URLs we are currently
    # interested in - those from the WashingtonPost have this nice property

    # OK - we got everything between the <article> and </article> tags, but that everything
    # includes a bunch of other stuff we don't want
    # Now - repeat, but this time we will only take what lies between <p> and </p> tags
    # these are HTML tags for "paragraph" i.e. this should give us the actual text of the article
    soup2 = BeautifulSoup(text)
    text = ' '.join(map(lambda p: p.text, soup2.find_all('p')))
    # use this code to get everything in that text that lies between a pair of 
    # <p> and </p> tags. We do this because we know that the URLs we are currently
    # interested in - those from the WashingtonPost have this nice property
    return soup.title.text, text
# Return a pair of values (article title, article body)
# Btw note that BeautifulSoup return the title without our doing anything special - 
# this is why BeautifulSoup works so much better than say regular expressions at parsing HTML


#####################################################################################
# OK! Now lets give this code a spin
#####################################################################################
someUrl = "https://www.washingtonpost.com/news/the-switch/wp/2015/08/06/why-kids-are-meeting-more-strangers-online-than-ever-before/"
# the article we would like to summarize
textOfUrl = get_only_text_washington_post_url(someUrl)
# get the title and text
fs = FrequencySummarizer()
# instantiate our FrequencySummarizer class and get an object of this class
summary = fs.summarize(textOfUrl[1], 3)
# get a summary of this article that is 3 sentences long


# In[ ]:



