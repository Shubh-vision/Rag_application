from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
load_dotenv()

# url = "https://www.deloitte.com/in/en/careers.html"
# url = "https://kpmg.com/in/en.html"
# url = "https://www.pwc.in/"

DEFAULT_URLS = [
    "https://kpmg.com/in/en.html" 
]

def load_doc(urls = None):

    urls = urls or DEFAULT_URLS

    all_docs = {}

    for url in urls:
        docs = WebBaseLoader(web_paths=[url],
                     header_template={
                    "User-Agent": "Mozilla/5.0"}).load()
        
        all_docs[url] = docs
    return all_docs


def get_docs(urls = None):
    return load_doc(urls)

    

  
