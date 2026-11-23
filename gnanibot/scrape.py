"""Step 1 (v2): Crawl gnani.ai with a real browser (Playwright), respecting
robots.txt, and save clean page text to data/pages.json."""
import json
import os
import re
import time
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura
from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

BASE = "https://www.gnani.ai"
ALLOWED_HOST = "gnani.ai"     # any host ending with this is in scope
BOT_NAME = "GnaniBotLearningProject"
USER_AGENT = f"{BOT_NAME}/1.0 (personal learning project)"
MAX_PAGES = 60                # safety cap
DELAY = 1.5                   # seconds between pages (be polite)
MIN_CHARS = 200               # skip near-empty pages
SKIP_EXT = (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".zip",
            ".mp4", ".mp3", ".css", ".js", ".xml", ".ico", ".woff", ".woff2")


def load_robots() -> RobotFileParser:
    """Fetch robots.txt ourselves (urllib's reader treats a 403 as 'block all')."""
    rp = RobotFileParser()
    try:
        r = requests.get(f"{BASE}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code == 200:
            rp.parse(r.text.splitlines())
            return rp
    except requests.RequestException:
        pass
    rp.parse([])  # nothing found -> nothing disallowed
    return rp


def normalize(url: str) -> str:
    url, _ = urldefrag(url)
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return f"{p.scheme}://{p.netloc}{path}"  # drops query string


def in_scope(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if not (p.netloc == ALLOWED_HOST or p.netloc.endswith("." + ALLOWED_HOST)):
        return False
    return not p.path.lower().endswith(SKIP_EXT)


def sitemap_urls(sitemap_url: str, depth: int = 0) -> list[str]:
    if depth > 2:
        return []
    try:
        r = requests.get(sitemap_url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return []
    except requests.RequestException:
        return []
    out = []
    for loc in re.findall(r"<loc>\s*(.*?)\s*</loc>", r.text):
        out.extend(sitemap_urls(loc, depth + 1) if loc.lower().endswith(".xml") else [loc])
    return out


def extract_text(html: str) -> str:
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if text:
        return text.strip()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()


def render(page, url: str) -> str:
    """Open the page in a real browser so JavaScript content appears."""
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
    except PWTimeout:
        pass  # some sites never go fully idle; use what has loaded
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")  # lazy content
    page.wait_for_timeout(1000)
    return page.content()


def crawl() -> list[dict]:
    robots = load_robots()
    queue = deque([normalize(BASE)])
    for u in sitemap_urls(f"{BASE}/sitemap.xml"):
        if in_scope(u):
            queue.append(normalize(u))

    seen, pages = set(), []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        # skip images/media/fonts: faster, and we only need text
        context.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ("image", "media", "font")
            else route.continue_(),
        )
        page = context.new_page()

        while queue and len(pages) < MAX_PAGES:
            url = queue.popleft()
            if url in seen:
                continue
            seen.add(url)

            if not robots.can_fetch(BOT_NAME, url):
                print(f"[robots] {url}")
                continue

            try:
                html = render(page, url)
            except Exception as e:
                print(f"[skip] {url} ({e})")
                continue

            text = extract_text(html)
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else url

            if len(text) >= MIN_CHARS:
                pages.append({"url": url, "title": title, "text": text})
                print(f"[ok {len(pages):>3}] {url}  ({len(text)} chars)")
            else:
                print(f"[thin] {url}")

            for a in soup.find_all("a", href=True):
                link = normalize(urljoin(url, a["href"]))
                if link not in seen and in_scope(link):
                    queue.append(link)

            time.sleep(DELAY)
        browser.close()
    return pages


if __name__ == "__main__":
    pages = crawl()
    os.makedirs("data", exist_ok=True)
    with open("data/pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(pages)} pages to data/pages.json")
