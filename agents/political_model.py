import datetime
from typing import Dict, List

EVENT_SEVERITY_MAP = {
    "scandal":           1.0,
    "protest":           0.6,
    "alliance":          0.3,
    "campaign activity": 0.1,
    "general":           0.0
}

def calculate_event_severity(event_tags: List[str]) -> float:
    """
    Takes a list of event tags and returns a combined severity score.
    Higher score = more politically volatile situation.
    """
    if not event_tags:
        return 0.0
    total = sum(EVENT_SEVERITY_MAP.get(tag, 0.0) for tag in event_tags)
    return round(total, 3)


def classify_political_risk(sentiment_score: float, severity: float) -> str:
    """
    Combines sentiment and event severity to produce a risk classification
    Returns: LOW, MODERATE, HIGH, CRITICAL 
    """
    if severity >= 1.5 and sentiment_score <= -0.5:
        return "CRITICAL"
    elif severity >= 0.8 or sentiment_score <= -0.4:
        return "HIGH"
    elif severity >= 0.3 or sentiment_score <= -0.1:
        return "MODERATE"
    else:
        return "LOW"

def analyze_political_signal(gemini_output: Dict) -> Dict:
    """
    This is the main entry point for the political model
    Takes raw Gemini Sentiment output and returns enriched political features.
    Feeds to TFT forecasting model
    """
    sentiment_score = gemini_output.get("sentiment_score",0.0)
    event_tags = gemini_output.get("event_tags",[])

    severity = calculate_event_severity(event_tags)
    risk_level = classify_political_risk(sentiment_score, severity)

    return {
        "constituency":     gemini_output.get("constituency"),
        "state":            gemini_output.get("state"),
        "date":             gemini_output.get("date"),
        "sentiment_score":  sentiment_score,
        "event_severity":   severity,
        "risk_level":       risk_level,
        "event_tags":       event_tags,
    }

if __name__ == "__main__":
    sample_gemini_output = {
        "constituency":    "Thiruvananthapuram",
        "state":           "Kerala",
        "date":            "2026-04-23",
        "sentiment_score": -0.4,
        "event_tags":      ["scandal", "protest"]

    }
    result = analyze_political_signal(sample_gemini_output)
    print("====== Political model sample output=====")
    for key,value in result.items():
        print(f" {key}: {value}")

