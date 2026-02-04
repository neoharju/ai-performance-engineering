"""AI assistant CLI commands wired to the unified PerformanceEngine."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from core.engine import get_engine


def _join_tokens(tokens: Optional[List[str]]) -> str:
    if not tokens:
        return ""
    return " ".join(tokens).strip()


def ask_question(args: Any) -> int:
    """Ask a performance question via the engine."""
    question = _join_tokens(getattr(args, "question", None))
    if not question:
        print("Question is required.")
        return 1
    include_citations = not getattr(args, "no_book", False)
    result = get_engine().ai.ask(question, include_citations=include_citations)
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0
    if result.get("success"):
        print(result.get("answer", ""))
        citations = result.get("citations") or []
        if citations:
            print("\nCitations:")
            for citation in citations:
                chapter = citation.get("chapter")
                section = citation.get("section")
                print(f"  - {chapter} {section}".strip())
        return 0
    print(result.get("error") or "Request failed.")
    return 1


def explain_concept(args: Any) -> int:
    """Explain a concept with citations via the engine."""
    concept = getattr(args, "concept", None) or ""
    if not concept.strip():
        print("Concept is required.")
        return 1
    result = get_engine().ai.explain(concept)
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0
    if result.get("success"):
        print(result.get("explanation", ""))
        key_points = result.get("key_points") or []
        if key_points:
            print("\nKey points:")
            for point in key_points:
                print(f"  - {point}")
        citations = result.get("citations") or []
        if citations:
            print("\nCitations:")
            for citation in citations:
                chapter = citation.get("chapter")
                section = citation.get("section")
                print(f"  - {chapter} {section}".strip())
        return 0
    print(result.get("error") or "Request failed.")
    return 1


def troubleshoot(args: Any) -> int:
    """Diagnose common issues."""
    issue = _join_tokens(getattr(args, "issue", None))
    if not issue:
        print("Issue description is required.")
        return 1
    result = get_engine().ai.troubleshoot(issue)
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0
    issues = result.get("issues") or []
    if not issues:
        print("No matching issues found.")
        return 0
    print(f"Found {len(issues)} issue(s):")
    for issue_entry in issues:
        title = issue_entry.get("title", "Unknown issue")
        severity = issue_entry.get("severity", "unknown")
        print(f"\n- {title} ({severity})")
        description = issue_entry.get("description")
        if description:
            print(f"  {description}")
        solutions = issue_entry.get("solutions") or []
        if solutions:
            print("  Fixes:")
            for solution in solutions:
                print(f"    • {solution}")
    return 0


def llm_status(args: Any) -> int:
    """Check LLM backend status."""
    result = get_engine().ai.status()
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0
    print(json.dumps(result, indent=2))
    return 0
