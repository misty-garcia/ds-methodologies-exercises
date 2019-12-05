import pandas as pd 
import numpy as np

from requests import get
from bs4 import BeautifulSoup
import os


def make_dictionary_from_article(url):
    headers = {'User-Agent': 'Codeup Ada Data Science'}
    response = get(url,headers=headers)
    soup = BeautifulSoup(response.text)
    title = soup.find("h1")
    body = soup.find("div", class_="mk-single-content")   
    
    return {
        "title": title.get_text(),
        "body": body.get_text()
    }


def get_blog_articles():
    # if we already have the data, read it locally
    if os.path.exists('articles.txt'):
        with open('articles.txt') as f:
            return f.read()

    # otherwise go fetch the data    
    urls = [
        "https://codeup.com/codeups-data-science-career-accelerator-is-here/",
        "https://codeup.com/data-science-myths/",
        "https://codeup.com/data-science-vs-data-analytics-whats-the-difference/",
        "https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/",
        "https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/",
    ]
    articles = []
    
    for url in urls:
        articles.append(make_dictionary_from_article(url))
 
    # save it for next time
#     with open('articles.txt', 'w') as f:
#         f.write(articles.txt)

    return articles