# app/agents/match_filter_agent.py

from typing import Dict, List, Any

def filter_matches_by_score(matches: Dict[str, List[Dict[str, Any]]], threshold: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filters the matches dictionary to only include items with score >= threshold.

    Args:
        matches (Dict[str, List[Dict[str, Any]]]): Matches from concept lookup.
        threshold (float): Minimum confidence score to retain a match.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Filtered matches.
    """
    filtered = {}

    for key, items in matches.items():
        high_confidence = [item for item in items if item.get("score", 0) >= threshold]
        if high_confidence:
            filtered[key] = high_confidence

    return filtered
