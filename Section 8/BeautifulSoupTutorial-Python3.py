
# coding: utf-8

# In[ ]:

from bs4 import BeautifulSoup

# Let's create an example html to play with 

html=['<html><heading style="font-size:20px"><i>This is the title<br><br></i></heading>',
     '<body><b>This is the body</b><p id="para1">This is para1<a href="www.google.com">Google</a></p>',
     '<p id="para2">This is para 2</p></body></html>']
html=''.join(html)
# This just creates one string out of the list above


# In[ ]:

# Instantiate a soup object. This automatically identifies a structure in the html and creates a parse tree
# you can navigate the structure/tree in the soup and extract pieces that you are interested in 
soup = BeautifulSoup(html)

# You can now print this html in a formatted, easy-to-read view
print(soup.prettify())


# In[ ]:

# At the top of the hierarchy in the parse tree is the <html></html> tag
# Then comes the <body></body> tag
# Within the body, the heading and paragraphs are 'siblings'
# The body is the parent of these tags and the html tag is the parent of the body tag 
# Each tag has attributes - name, contents (a list), text, parent and siblings 

# name attribute is just the name of the tag 
soup.html.name


# In[ ]:

soup.body.name


# In[ ]:

# text attribute will mush together all the text in all the children of that tag
soup.body.text


# In[ ]:

# contents is a list of the children of that tag
# In our example, the html tag has only 1 child, the body tag has 4 children 
soup.html.contents


# In[ ]:

soup.body.contents


# In[ ]:

# parent and siblings referencing helps you navigate the parse tree
soup.body.parent.name


# In[ ]:

soup.b.nextSibling


# In[ ]:

soup.p.previousSibling


# In[ ]:

# findAll, find are methods to search the tree for specific tags, or tags with certain attributes 

bold = soup.findAll('b')
# This will find all the tags which have the text in bold (enclosed in <b></b> tags) and return a list
print(bold)


# In[ ]:

# to extract only the text, take each element of this list and get the text attribute
print(bold[0].text)


# In[ ]:

# Let's get all the text that is in the paragraphs (enclosed in <p></p> tags) as a single string 
paras = ' '.join([p.text for p in soup.findAll('p')])
print(paras)


# In[ ]:

# findAll can look for attributes as well. Let's find the text for the paragraph with id 'para2'
soup.findAll(id="para2")[0].text
#soup.findAll gives us a list, we pick the first element (there is only 1 element in this case) and print the text


# In[ ]:

# Let's find any text with font size 20

font20=' '.join([p.text for p in soup.findAll(style="font-size:20px")])
print(font20)


# In[ ]:

# You can also pass in a list or a dictionary of tag names to search for 
soup.findAll(['b','p'])
soup.findAll({'b':True,'p':True})


# In[ ]:

# Let's see how we can find links. This is super-useful - say you want to find all the links on a page and iterate through
# them to do some scraping for each of those links 
links = soup.find('a')

# Links are generally of the form <a href='link'>'link-text'</a>
print(links)


# In[ ]:

# Notice how we used find instead of findAll - this will just give us the first tag that matches the search, in this
# case we have only 1 link on our page. soup.findAll will return a list of links and you can limit the number of 
# results using the limit keyword soup.findAll('a',limit=10)


# In[ ]:

# Let's extract the url and the text separately 
print(links['href']+" is the url and "+links.text+" is the text")


# In[ ]:

# find can navigate the parse tree as well. findParents, findNextSiblings,findPreviousSiblings all work 
# similar to findAll, but will search only within those branches of the tree. 
# findNext, findPrevious and findAllNext and findAllPrevious can be used to find matches starting from 
# a specified point. 

# Let's say you want the text of the first paragraph after the first occurrence of the text "Google" 

soup.find(text="Google").findNext('p').text


# In[ ]:

# A little shortcut to using findAll - if you call the tag itself as a function, you can use it in place of findAll
# with the same arguments 

soup.body('p')


# In[ ]:

soup.findAll('p')


# In[ ]:

#BeautifulSoup makes parsing html or xml very intuitive and elegant. Doing the same thing with regular expressions 
# is prone to leading you to pulling your hair out :) In most situations for screen-scraping projects, BeautifulSoup
# is a life-saver! 


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



