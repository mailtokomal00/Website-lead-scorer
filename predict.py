"""
predict.py
Website Visitor Lead Conversion Scorer - Inference & Scoring Engine
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import joblib
import pandas as pd
import numpy as np

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lead_conversion_model.joblib")

_MODEL_CACHE = None

def load_lead_model(model_path=MODEL_PATH):
    """Load and cache the trained Scikit-learn Pipeline."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python train.py' to generate data and train the model first."
        )
    
    _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE

def classify_lead_score(score: float):
    """
    Classify lead score into predefined business tiers:
      - High Lead: 70 - 100
      - Medium Lead: 40 - 69
      - Low Lead: 0 - 39
    """
    if score >= 70.0:
        return {
            "tier": "High Lead",
            "badge_color": "#10B981", # Emerald Green
            "badge_bg": "#D1FAE5",
            "priority": "P1 - Critical",
            "recommended_action": "Direct Sales Outreach: Assign directly to senior sales rep for personalized demo/call within 1 hour.",
            "qualification_status": "Sales Qualified Lead (SQL)"
        }
    elif score >= 40.0:
        return {
            "tier": "Medium Lead",
            "badge_color": "#F59E0B", # Amber
            "badge_bg": "#FEF3C7",
            "priority": "P2 - Warm",
            "recommended_action": "Automated Email Sequence: Enroll in product-led nurture campaign with tailored case studies and webinar invites.",
            "qualification_status": "Marketing Qualified Lead (MQL)"
        }
    else:
        return {
            "tier": "Low Lead",
            "badge_color": "#EF4444", # Red
            "badge_bg": "#FEE2E2",
            "priority": "P3 - Informational",
            "recommended_action": "Passive Remarketing: Deliver retargeting display ads and invite to join newsletter or community.",
            "qualification_status": "Top of Funnel (Prospect)"
        }

def predict_lead(
    session_duration: float,
    pages_visited: int,
    referral_source: str,
    content_downloads: int,
    model_path: str = MODEL_PATH
) -> dict:
    """
    Predict lead conversion probability and calculate comprehensive lead scoring metrics.
    
    Parameters:
      session_duration (float or int): Visitor session duration in seconds.
      pages_visited (int): Number of unique pages browsed.
      referral_source (str): Channel source ('Google', 'Social Media', 'Direct', 'Email', 'Referral').
      content_downloads (int): Number of downloads (whitepapers, decks, brochures).
      
    Returns:
      dict: Scoring metrics including probability, lead_score, category, classification tier, and insights.
    """
    # Validate and clean inputs
    valid_sources = ['Google', 'Social Media', 'Direct', 'Email', 'Referral']
    clean_source = str(referral_source).strip()
    if clean_source not in valid_sources:
        # Default or fallback match
        matched = [s for s in valid_sources if s.lower() == clean_source.lower()]
        if matched:
            clean_source = matched[0]
        else:
            clean_source = 'Direct'
            
    session_duration = max(0.0, float(session_duration))
    pages_visited = max(1, int(pages_visited))
    content_downloads = max(0, int(content_downloads))
    
    # Format input DataFrame matching the pipeline expected column names
    input_df = pd.DataFrame([{
        'session_duration': session_duration,
        'pages_visited': pages_visited,
        'referral_source': clean_source,
        'content_downloads': content_downloads
    }])
    
    # Load model and predict
    model = load_lead_model(model_path)
    predicted_class = int(model.predict(input_df)[0])
    predicted_probs = model.predict_proba(input_df)[0]
    conversion_prob = float(predicted_probs[1]) # probability of class 1 (Converted)
    
    # Lead score from 0 to 100 based on predicted probability
    lead_score = round(conversion_prob * 100.0, 1)
    
    # Category tier
    tier_info = classify_lead_score(lead_score)
    
    # Feature impact breakdown
    drivers = []
    if content_downloads >= 2:
        drivers.append(f"Strong intent signal: {content_downloads} content downloads")
    elif content_downloads == 0:
        drivers.append("No content asset downloads yet")
        
    if session_duration >= 300:
        drivers.append(f"Deep engagement: {int(session_duration)}s active browsing")
    elif session_duration < 90:
        drivers.append("Short session duration indicates exploratory visit")
        
    if pages_visited >= 6:
        drivers.append(f"High multi-page exploration ({pages_visited} pages)")
        
    if clean_source in ['Referral', 'Email']:
        drivers.append(f"High-converting referral channel ({clean_source})")
    elif clean_source == 'Social Media':
        drivers.append("Social traffic typically exhibits lower initial conversion")

    return {
        "lead_score": lead_score,
        "conversion_probability": round(conversion_prob * 100.0, 2),
        "raw_probability": round(conversion_prob, 4),
        "predicted_class": predicted_class,
        "category": tier_info["tier"],
        "priority": tier_info["priority"],
        "qualification_status": tier_info["qualification_status"],
        "recommended_action": tier_info["recommended_action"],
        "badge_color": tier_info["badge_color"],
        "badge_bg": tier_info["badge_bg"],
        "input_features": {
            "session_duration": session_duration,
            "pages_visited": pages_visited,
            "referral_source": clean_source,
            "content_downloads": content_downloads
        },
        "key_drivers": drivers
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict Website Visitor Lead Conversion Score")
    parser.add_argument("--duration", type=float, default=320, help="Session duration in seconds")
    parser.add_argument("--pages", type=int, default=6, help="Pages visited")
    parser.add_argument("--source", type=str, default="Google", choices=['Google', 'Social Media', 'Direct', 'Email', 'Referral'])
    parser.add_argument("--downloads", type=int, default=2, help="Content downloads")
    
    args = parser.parse_args()
    
    print("\n--- Running Lead Score Inference ---")
    result = predict_lead(
        session_duration=args.duration,
        pages_visited=args.pages,
        referral_source=args.source,
        content_downloads=args.downloads
    )
    print(f"Lead Score:             {result['lead_score']} / 100")
    print(f"Conversion Probability: {result['conversion_probability']}%")
    print(f"Category:               {result['category']}")
    print(f"Qualification:          {result['qualification_status']}")
    print(f"Recommended Action:     {result['recommended_action']}")
    print("------------------------------------\n")
