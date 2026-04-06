import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin, urlparse

# ==============================
# CONFIG
# ==============================

BASE_URL = "https://www.cuchd.in"
DOMAIN = "cuchd.in"

MAX_PAGES = 150   # limit crawling
TIMEOUT = 10

OUTPUT_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VISITED = set()


# ==============================
# CLEAN FILE NAME
# ==============================

def clean_filename(url):
    name = url.replace("https://", "").replace("http://", "")
    name = name.replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)

    if len(name) > 150:
        name = name[:150]

    return name + ".md"


# ==============================
# CLEAN TEXT
# ==============================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


# ==============================
# SAVE PAGE
# ==============================

def save_page(url, text):

    filename = clean_filename(url)
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Source: {url}\n\n")
        f.write(text)

    print("Saved:", filename)


# ==============================
# CHECK INTERNAL LINK
# ==============================

def is_internal_link(url):

    parsed = urlparse(url)

    if parsed.netloc == "":
        return True

    return DOMAIN in parsed.netloc


# ==============================
# SCRAPE PAGE
# ==============================

def scrape_page(url):

    if url in VISITED:
        return

    if len(VISITED) >= MAX_PAGES:
        return

    VISITED.add(url)

    print("Scraping:", url)

    try:
        response = requests.get(url, timeout=TIMEOUT)

        if response.status_code != 200:
            return

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts/styles
        # Remove scripts/styles/navigation
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.extract()

        text = soup.get_text(separator=" ")

        text = clean_text(text)

        if len(text) > 200:
            save_page(url, text)

        # Crawl links
        for link in soup.find_all("a", href=True):

            href = link["href"]
            next_url = urljoin(url, href)

            if is_internal_link(next_url):
                scrape_page(next_url)

    except Exception as e:
        print("Error:", e)


# ==============================
# START CRAWLING
# ==============================

if __name__ == "__main__":

    print("\nStarting website crawl...\n")

    scrape_page(BASE_URL)

    print("\nCrawling finished")
    print("Total pages scraped:", len(VISITED))