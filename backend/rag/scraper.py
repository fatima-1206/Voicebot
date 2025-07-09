import requests
from bs4 import BeautifulSoup
import os
import json
from urllib.parse import urljoin
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Base URL of your university website

# Store scraped documents
scraped_docs = []

# Keep track of visited URLs to avoid re-scraping
visited_urls = set()
scraped_data_file = "cli_app/scraped_docs.json"

# Function to fetch page content
def fetch_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# Function to extract text from the HTML content of a page
def extract_text_from_page(html):
    soup = BeautifulSoup(html, 'html.parser', features="lxml")
    # Extract text from paragraphs and divs, ignoring scripts and styles
    texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'div'])
    content = " ".join([text.get_text(strip=True) for text in texts if text.get_text(strip=True)])
    return content

# Function to crawl and scrape the website
def crawl_website(start_url:str, excluded_sublinks:str="", depth:int=-1):
    pages_not_to_visit = []
    if excluded_sublinks != "":
        pages_not_to_visit = excluded = excluded_sublinks.split("|")
    pages_to_visit = [start_url]
    print(f"Scraping {start_url} and sub pages...")
    while pages_to_visit:
        url = pages_to_visit.pop()
        
        if url in visited_urls:
            continue
        
        visited_urls.add(url)

        
        html = fetch_page(url)
        if not html:
            continue
        
        # Extract content
        page_text = extract_text_from_page(html)
        if page_text:
            # Save the extracted text into the scraped_docs list
            scraped_docs.append({
                "url": url,
                "content": page_text
            })
        
        # Find all links on the current page and add to the list to visit
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a', href=True)
        for link in links:
            link_url = urljoin(url, link['href'])
            # Only add links from the same domain
            if start_url in link_url and link_url not in visited_urls:
                # check if there are some words in the url to filter out useless pages
                if any(word.lower() in link_url.lower() for word in pages_not_to_visit):
                    continue
                else:
                    pages_to_visit.append(link_url)

    return scraped_docs

# Function to save scraped documents to a JSON file
def save_to_json(docs, filename=scraped_data_file):
    with open(filename, 'a', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(docs)} documents to {filename}")

# Function to store documents in FAISS index
def store_in_faiss(docs):
    embeddings = OpenAIEmbeddings()  # You can replace this with your preferred embeddings model
    faiss_index = FAISS.from_documents(docs, embeddings)
    faiss_index.save_local("university_knowledge_base")
    print("Stored documents in FAISS.")

