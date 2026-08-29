from __future__ import annotations

import asyncio
import os
from typing import List

import requests

def search_web(query: str, max_results: int = 5) -> dict:
    """Search the web for information about an art-related query.

    Args:
        query: The search query string (e.g. 'Who painted the Mona Lisa?').
        max_results: Maximum number of results to return.

    Returns:
        A dict with 'summary' (str | None) and 'results' (list of dicts
        with 'title', 'url', 'snippet').
    """
    tavily_key = os.getenv("TAVILY_API_KEY")

    if tavily_key:
        try:
            result = _tavily_search(tavily_key, query, max_results)
            if result["results"]:
                return result
        except Exception:
            pass

    return _ddg_search(query, max_results)


def _tavily_search(api_key: str, query: str, max_results: int) -> dict:
    """Call the Tavily search API."""
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
    }
    resp = requests.post(url, json=payload, timeout=15) # Fetch req using Python
    resp.raise_for_status()
    data = resp.json()

    summary = data.get("answer")
    results: List[dict] = []
    for r in data.get("results", [])[:max_results]:
        results.append(
            {
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "snippet": (r.get("content", "") or "")[:220],
            }
        )
    return {"summary": summary, "results": results}

def _ddg_search(query: str, max_results: int) -> dict:
    """Call DuckDuckGo Instant Answer API (no key required)."""
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
        "no_redirect": 1,
    }
    resp = requests.get(url, params=params, timeout=15)

    return resp
    if resp.status_code != 200:
        return {"summary": None, "results": []}

    data = resp.json()
    summary = data.get("AbstractText")

    results: List[dict] = []
    candidates: list = []
    if isinstance(data.get("Results"), list):
        candidates += data["Results"]
    if isinstance(data.get("RelatedTopics"), list):
        candidates += _flatten_topics(data["RelatedTopics"])

    for item in candidates:
        if len(results) >= max_results:
            break
        page_url = item.get("FirstURL") or item.get("URL") or ""
        text = item.get("Text") or item.get("Result") or ""
        title = item.get("Title") or (text.split(" - ")[0] if " - " in text else "Result")
        snippet = text.strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        if page_url:
            results.append({"title": title, "url": page_url, "snippet": snippet})

    return {"summary": summary, "results": results}


def _flatten_topics(topics: list) -> list:
    """Recursively flatten nested DuckDuckGo topic groups."""
    items: list = []
    for item in topics:
        if "FirstURL" in item:
            items.append(item)
        if "Topics" in item and isinstance(item["Topics"], list):
            items.extend(_flatten_topics(item["Topics"]))
    return items