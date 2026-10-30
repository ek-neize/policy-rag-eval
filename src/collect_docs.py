"""
Step 1: Collect FCAC consumer guidance pages.

What this script does:
  1. Starts from a few "seed" pages on the FCAC website.
  2. Follows links that stay inside FCAC's consumer guidance section.
  3. For each page, keeps only the main article text (no menus or footers).
  4. Saves the text as a .md file and records its metadata in manifest.csv.

It is polite to the website: it checks robots.txt, identifies itself,
waits between requests, and stops after a fixed number of pages.
"""

import csv
import re
import time
from collections import Counter, deque
from datetime import date
from pathlib import Path
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Settings (change these to collect more or fewer pages)
# ---------------------------------------------------------------------------
SITE = "https://www.canada.ca"
SECTION = "/en/financial-consumer-agency/services/"   # only pages under here

SEEDS = [
    "credit-cards.html",
    "banking.html",
    "mortgages.html",
    "loans.html",
    "credit-reports-score/order-credit-report.html",
    "banking/bank-accounts.html",
    "banking/opening-bank-account.html",
    "rights-responsibilities/rights-mortgages/financial-difficulties.html",
    "interest-rates-rise.html",
]

EXCLUDED_TOPICS = {"financial-toolkit"}  # a long course; would crowd out other topics
MAX_DOCS = 50          # stop after saving this many documents
MAX_PER_TOPIC = 8      # keeps the mix balanced across topics
MAX_REQUESTS = 150     # safety limit on total downloads
MIN_WORDS = 150        # pages shorter than this are menus/index pages, not articles
DELAY_SECONDS = 1.0    # pause between requests (politeness)

USER_AGENT = "policy-rag-eval/0.1 (student portfolio project; https://github.com/ek-neize/policy-rag-eval)"

OUT_DIR = Path("data/raw")
DOCS_DIR = OUT_DIR / "docs"


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------
def normalize(url: str) -> str:
    """Remove #fragments and ?queries so the same page isn't counted twice."""
    url, _fragment = urldefrag(url)
    parts = urlparse(url)
    return f"https://{parts.netloc}{parts.path}"


def in_section(url: str) -> bool:
    """True if the URL is an English FCAC consumer guidance page."""
    parts = urlparse(url)
    return (
        parts.netloc == "www.canada.ca"
        and parts.path.startswith(SECTION)
        and parts.path.endswith(".html")
    )


def topic_of(url: str) -> str:
    """'.../services/banking/bank-accounts.html' -> 'banking'."""
    rest = urlparse(url).path[len(SECTION):]
    return rest.split("/")[0].removesuffix(".html")


def doc_id_of(url: str) -> str:
    """A short, readable, stable ID used later in citations."""
    rest = urlparse(url).path[len(SECTION):].removesuffix(".html")
    return rest.replace("/", "__")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def html_to_text(main) -> str:
    """Turn the page's <main> HTML into clean text, keeping headings as markdown."""
    # Remove HTML comments.
    for comment in main.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # Collapse the messy whitespace that comes from HTML source formatting.
    for s in main.find_all(string=True):
        s.replace_with(re.sub(r"\s+", " ", s))

    # Line breaks (<br>) become real line breaks, so words on either side don't merge.
    for br in main.find_all("br"):
        br.replace_with("\n")

    # Mark headings (# Title, ## Section) so chunking can use them in Step 2.
    for level in range(1, 5):
        for h in main.find_all(f"h{level}"):
            h.insert_before("\n\n" + "#" * level + " ")
            h.append("\n")

    # Mark list items and table cells; end block elements with a line break.
    for li in main.find_all("li"):
        li.insert_before("\n- ")
    for cell in main.find_all(["td", "th"]):
        cell.append(" | ")
    for block in main.find_all(["p", "div", "section", "tr", "dt", "dd", "ul", "ol", "table"]):
        block.append("\n")

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in main.get_text().splitlines()]
    lines = [line for line in lines if line not in {"-", "|"}]
    text = "\n".join(lines)
    # Remove invisible characters: zero-width space, soft hyphen, byte-order mark.
    text = re.sub("[\u200b\u00ad\ufeff]", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_document(soup: BeautifulSoup, url: str):
    """Pull the title, metadata and main text out of one downloaded page."""
    main = soup.find("main")
    if main is None:
        return None

    def meta(name: str) -> str:
        tag = soup.find("meta", attrs={"name": name})
        return tag["content"].strip() if tag and tag.get("content") else ""

    h1 = main.find("h1")
    title = meta("dcterms.title") or (h1.get_text(" ", strip=True) if h1 else url)

    # Drop parts of <main> that aren't the article itself.
    for junk in main.select("script, style, noscript, nav, .pagedetails, #wb-dtmd"):
        junk.extract()

    text = html_to_text(main)
    return {
        "doc_id": doc_id_of(url),
        "title": title,
        "topic": topic_of(url),
        "url": url,
        "date_modified": meta("dcterms.modified"),
        "retrieved_on": date.today().isoformat(),
        "word_count": len(text.split()),
        "description": meta("description"),
        "text": text,
    }


# ---------------------------------------------------------------------------
# Main crawl loop
# ---------------------------------------------------------------------------
def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    robots = robotparser.RobotFileParser(SITE + "/robots.txt")
    robots.read()

    queue = deque(normalize(SITE + SECTION + seed) for seed in SEEDS)
    seen = set(queue)
    saved, saved_urls = [], set()
    per_topic = Counter()
    requests_made = 0

    while queue and len(saved) < MAX_DOCS and requests_made < MAX_REQUESTS:
        url = queue.popleft()
        topic = topic_of(url)
        if topic in EXCLUDED_TOPICS or per_topic[topic] >= MAX_PER_TOPIC:
            continue
        if not robots.can_fetch(USER_AGENT, url):
            print(f"  robots.txt says no: {url}")
            continue

        try:
            response = session.get(url, timeout=20)
        except requests.RequestException as error:
            print(f"  ! failed: {url} ({error})")
            continue
        finally:
            requests_made += 1
            time.sleep(DELAY_SECONDS)

        if response.status_code != 200 or "text/html" not in response.headers.get("Content-Type", ""):
            print(f"  ! skipped (HTTP {response.status_code}): {url}")
            continue

        final_url = normalize(response.url)          # in case of a redirect
        if not in_section(final_url) or final_url in saved_urls:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Discover new pages to visit from this page's links.
        for a in soup.find_all("a", href=True):
            link = normalize(urljoin(final_url, a["href"]))
            if in_section(link) and link not in seen:
                seen.add(link)
                queue.append(link)

        doc = extract_document(soup, final_url)
        if doc is None or doc["word_count"] < MIN_WORDS:
            print(f"  · index page, used for links only: {final_url}")
            continue
        if "## Services and information" in doc["text"]:
            print(f"  · topic menu page, used for links only: {final_url}")
            continue
        if per_topic[doc["topic"]] >= MAX_PER_TOPIC:
            continue

        (DOCS_DIR / f"{doc['doc_id']}.md").write_text(doc["text"], encoding="utf-8")
        saved.append(doc)
        saved_urls.add(final_url)
        per_topic[doc["topic"]] += 1
        print(f"  ✓ [{doc['topic']}] {doc['title']} ({doc['word_count']} words)")

    # Write the manifest: one row of metadata per saved document.
    fields = ["doc_id", "title", "topic", "url", "date_modified", "retrieved_on", "word_count", "description"]
    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(saved)

    print(f"\nSaved {len(saved)} documents using {requests_made} requests.")
    print("Documents per topic:")
    for topic, count in per_topic.most_common():
        print(f"  {topic:35s} {count}")
    print(f"Total words: {sum(d['word_count'] for d in saved):,}")


if __name__ == "__main__":
    main()
