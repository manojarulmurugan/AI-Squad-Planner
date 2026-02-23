#!/usr/bin/env python3
"""
AI Trip Planner - Complete Standalone Script
============================================
This script implements a complete AI-powered trip planning system using LangChain
and LangGraph to plan fair, budget-aware 2-day group trips.

Requirements:
- Set GOOGLE_API_KEY environment variable
- Set SERPAPI_KEY environment variable (for flights and hotels)
- activities.csv file in the same directory
"""

import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import Counter

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    # User can set environment variables manually
    pass

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# ============================================================================
# SECTION 1: SAMPLE DATA & GROUP SETUP
# ============================================================================

# 5 fake travelers with different tastes & budgets, origin cities, and date preferences
group_members = [
    {
        "id": "u1",
        "name": "Alice",
        "origin_city": "ORD",  # Chicago O'Hare
        "total_budget": 9000,
        "preference_weights": {
            "nightlife": 1,
            "adventure": 5,
            "shopping": 1,
            "food": 1,
            "urban": 5,
        },
        "notes": "Budget foodie, hates early mornings, loves walkable neighborhoods and speakeasies.",
        "preferred_windows": [
            {"trip_start": "2025-12-20", "trip_end": "2025-12-21"},  # Saturday-Sunday
            {"trip_start": "2025-12-27", "trip_end": "2025-12-28"},  # Saturday-Sunday
            {"trip_start": "2026-01-03", "trip_end": "2026-01-04"}   # Saturday-Sunday
        ]
    },
    {
        "id": "u2",
        "name": "Ben",
        "origin_city": "JFK",  # New York JFK
        "total_budget": 7000,
        "preference_weights": {
            "nightlife": 2,
            "adventure": 5,
            "shopping": 1,
            "food": 1,
            "urban": 4,
        },
        "notes": "Loves outdoor activities and casual spots, not into heavy nightlife.",
        "preferred_windows": [
            {"trip_start": "2025-12-20", "trip_end": "2025-12-21"},  # Saturday-Sunday (common)
            {"trip_start": "2025-12-13", "trip_end": "2025-12-14"},  # Saturday-Sunday
            {"trip_start": "2026-01-10", "trip_end": "2026-01-11"}   # Saturday-Sunday
        ]
    },
    {
        "id": "u3",
        "name": "Carla",
        "origin_city": "LAX",  # Los Angeles
        "total_budget": 6500,
        "preference_weights": {
            "nightlife": 1,
            "adventure": 5,
            "shopping": 4,
            "food": 1,
            "urban": 5,
        },
        "notes": "Enjoys shopping districts and museums, prefers quieter evenings.",
        "preferred_windows": [
            {"trip_start": "2025-12-20", "trip_end": "2025-12-21"},  # Saturday-Sunday (common)
            {"trip_start": "2025-12-27", "trip_end": "2025-12-28"},  # Saturday-Sunday
            {"trip_start": "2026-01-03", "trip_end": "2026-01-04"}   # Saturday-Sunday
        ]
    },
    {
        "id": "u4",
        "name": "Deepak",
        "origin_city": "SFO",  # San Francisco
        "total_budget": 700,
        "preference_weights": {
            "nightlife": 1,
            "adventure": 5,
            "shopping": 2,
            "food": 1,
            "urban": 5,
        },
        "notes": "On a tighter budget, okay with simple food, likes city walks and parks.",
        "preferred_windows": [
            {"trip_start": "2025-12-20", "trip_end": "2025-12-21"},  # Saturday-Sunday (common - 3rd person)
            {"trip_start": "2025-12-13", "trip_end": "2025-12-14"},  # Saturday-Sunday
            {"trip_start": "2026-01-10", "trip_end": "2026-01-11"}   # Saturday-Sunday
        ]
    },
    {
        "id": "u5",
        "name": "Emma",
        "origin_city": "ATL",  # Atlanta
        "total_budget": 1100,
        "preference_weights": {
            "nightlife": 1,
            "adventure": 5,
            "shopping": 1,
            "food": 2,
            "urban": 5,
        },
        "notes": "Happy to splurge a bit on nightlife and good food.",
        "preferred_windows": [
            {"trip_start": "2025-12-27", "trip_end": "2025-12-28"},  # Saturday-Sunday
            {"trip_start": "2026-01-03", "trip_end": "2026-01-04"},  # Saturday-Sunday
            {"trip_start": "2026-01-10", "trip_end": "2026-01-11"}   # Saturday-Sunday
        ]
    },
]

DIMS = ["nightlife", "adventure", "shopping", "food", "urban"]

# ============================================================================
# SECTION 2: HELPER FUNCTIONS
# ============================================================================

def normalize_weights(w_dict, dims):
    """Normalize preference weights to sum to 1.0"""
    total = sum(w_dict[d] for d in dims)
    if total == 0:
        return {d: 1.0 / len(dims) for d in dims}
    return {d: w_dict[d] / total for d in dims}


def find_most_common_window(group_members):
    """
    Find the most common two-day window from all group members' preferred windows.
    
    Args:
        group_members: List of member dicts, each with a "preferred_windows" list
                      containing dicts with "trip_start" and "trip_end" dates
    
    Returns:
        dict with "trip_start" and "trip_end" of the most common window,
        or None if no windows found
    """
    # Collect all preferred windows from all members
    window_counts = Counter()
    
    for member in group_members:
        preferred_windows = member.get("preferred_windows", [])
        for window in preferred_windows:
            # Create a tuple key from the dates for counting
            window_key = (window.get("trip_start"), window.get("trip_end"))
            if window_key[0] and window_key[1]:  # Ensure both dates exist
                window_counts[window_key] += 1
    
    if not window_counts:
        return None
    
    # Find the most common window
    most_common = window_counts.most_common(1)[0]
    trip_start, trip_end = most_common[0]
    count = most_common[1]
    
    return {
        "trip_start": trip_start,
        "trip_end": trip_end,
        "member_count": count,  # How many members prefer this window
        "total_members": len(group_members)
    }


def compute_fairness_summary(group_members):
    """
    Calculate fairness score based on budget ratios.
    Each member dict must have: name, total_budget, flight_price, hotel_share, activities_spend
    """
    def affordability_component(mean_ratio):
        if mean_ratio <= 0.6:
            return 1.0
        if mean_ratio >= 1.4:
            return 0.0
        return 1.0 - (mean_ratio - 0.6) / (1.4 - 0.6)

    def equality_component(std_ratio):
        if std_ratio <= 0.05:
            return 1.0
        if std_ratio >= 0.25:
            return 0.0
        return 1.0 - (std_ratio - 0.05) / (0.25 - 0.05)

    def affordability_label(r):
        if r <= 0.6:
            return "comfortable"
        elif r <= 1.0:
            return "stretch"
        elif r <= 1.3:
            return "risky"
        else:
            return "not_recommended"

    ratios = []
    per_person = []

    for p in group_members:
        trip_cost = p["flight_price"] + p["hotel_share"] + p["activities_spend"]
        r = trip_cost / p["total_budget"]
        label = affordability_label(r)

        per_person.append({
            "name": p["name"],
            "trip_cost": round(trip_cost, 2),
            "budget": p["total_budget"],
            "affordability_ratio": round(r, 3),
            "affordability_label": label,
        })
        ratios.append(min(r, 2.0))

    ratios = np.array(ratios, dtype=float)
    mean_ratio = float(ratios.mean())
    std_ratio = float(ratios.std())

    aff_comp = affordability_component(mean_ratio)
    eq_comp = equality_component(std_ratio)

    fairness_score = 100 * (0.6 * aff_comp + 0.4 * eq_comp)
    fairness_score = max(0.0, min(100.0, fairness_score))

    return {
        "trip_fairness_score": round(fairness_score, 1),
        "mean_ratio": round(mean_ratio, 3),
        "std_ratio": round(std_ratio, 3),
        "per_person": per_person,
    }


# Load activities dataframe and define constants
activities_df = pd.read_csv("activities.csv")
TAG_COLS = [f"tag_{d}" for d in DIMS]


def compute_base_scores_for_city(
    city: str,
    group_members: List[Dict[str, Any]],
    top_k_activities: int = 10,
    min_stars: float = 3.5,
    min_review_count: int = 30,
) -> Dict[str, Any]:
    """
    Filter activities for the city, compute base enjoyment/value scores per user and per activity,
    and return the top K activities by group value.
    """
    # 1) Normalize user preference weights
    for m in group_members:
        m["norm_weights"] = normalize_weights(m["preference_weights"], DIMS)

    # 2) Filter activities for the chosen city & quality
    city_df = activities_df[
        (activities_df["city"] == city)
        & (activities_df["stars"] >= min_stars)
        & (activities_df["review_count"] >= min_review_count)
    ].copy()

    if city_df.empty:
        raise ValueError(f"No activities found for city={city} with current filters.")

    # Ensure price_proxy exists; if not, create from price_level
    if "price_proxy" not in city_df.columns:
        price_map = {1: 15, 2: 30, 3: 50, 4: 80}
        city_df["price_proxy"] = city_df["price_level"].map(price_map)

    # 3) Compute base enjoyment + value per user and group
    group_values = []
    per_user_scores: Dict[str, Dict[str, float]] = {
        m["id"]: {} for m in group_members
    }  # user_id -> {business_id: value_score}

    for idx, row in city_df.iterrows():
        biz_id = row["business_id"]
        tags_vec = np.array([row[col] for col in TAG_COLS], dtype=float)
        price = float(row["price_proxy"]) if not pd.isna(row["price_proxy"]) else 1.0

        user_values = []
        for m in group_members:
            w_vec = np.array([m["norm_weights"][d] for d in DIMS], dtype=float)
            base_enjoyment = float((w_vec * tags_vec).sum())
            value_score = base_enjoyment / max(price, 1e-6)
            per_user_scores[m["id"]][biz_id] = value_score
            user_values.append(value_score)

        group_value = float(np.mean(user_values))
        group_values.append((biz_id, group_value))

    # 4) Select top K activities by group value
    group_values_sorted = sorted(group_values, key=lambda x: x[1], reverse=True)
    top_biz_ids = [biz_id for biz_id, _ in group_values_sorted[:top_k_activities]]

    top_city_df = city_df[city_df["business_id"].isin(top_biz_ids)].copy()

    return {
        "city_df": top_city_df,
        "per_user_scores": per_user_scores,
        "top_biz_ids": top_biz_ids,
    }


def extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON from markdown code blocks or text that may contain extra content.
    Handles cases where LLM wraps JSON in ```json ... ``` or ``` ... ```
    Also handles cases where there's extra text before/after the JSON.
    Prefers the largest/most complete JSON block if multiple are found.
    """
    import re
    
    content = text.strip()
    
    # First, try to find ALL JSON markdown code blocks
    # Extract everything between ```json and ``` (or ``` and ```)
    # Use a pattern that captures everything between the fences
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    json_matches = re.finditer(json_block_pattern, content, re.DOTALL)
    
    json_blocks = []
    for match in json_matches:
        block_content = match.group(1).strip()
        # Only keep blocks that look like JSON (start with {)
        if block_content.startswith('{'):
            json_blocks.append(block_content)
    
    if json_blocks:
        # If multiple blocks found, prefer the longest one (likely the complete response)
        if len(json_blocks) > 1:
            # Return the longest block
            longest_block = max(json_blocks, key=len)
            return longest_block.strip()
        else:
            return json_blocks[0].strip()
    
    # If no markdown blocks, try to find JSON object directly - match balanced braces
    # Start from the first { and find the matching }
    brace_count = 0
    start_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    return content[start_idx:i+1].strip()
    
    # If no balanced JSON found, try to find first { ... } pattern
    first_brace = content.find('{')
    if first_brace != -1:
        # Try to parse from first brace to end, then trim
        potential_json = content[first_brace:]
        # Try to find the last } that might close it
        last_brace = potential_json.rfind('}')
        if last_brace != -1:
            return potential_json[:last_brace+1].strip()
    
    # If no JSON found, return original (will fail gracefully in caller)
    return content.strip()


def build_llm_for_reranking(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Create a Gemini LLM instance for reranking."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=temperature,
    )
    return llm


def rerank_with_gemini(
    city: str,
    group_members: List[Dict[str, Any]],
    base_info: Dict[str, Any],
    llm: ChatGoogleGenerativeAI,
) -> List[Dict[str, Any]]:
    """
    For each user and each of the top activities, call Gemini to get:
      - adjusted_enjoyment_score
      - recommended_time_of_day
      - note

    Returns a list of dicts: one entry per (user, activity).
    """
    city_df = base_info["city_df"]
    per_user_scores = base_info["per_user_scores"]
    top_biz_ids = base_info["top_biz_ids"]

    # We'll process per-user, sending all top activities in one call per user
    results: List[Dict[str, Any]] = []

    for m in group_members:
        # Handle missing total_budget gracefully (use default or get from budget field)
        total_budget = m.get("total_budget") or m.get("budget")
        if total_budget is None:
            raise ValueError(f"Missing 'total_budget' field for user {m.get('id', 'unknown')}. Each group member must have a 'total_budget' field.")
        
        user_profile = {
            "id": m["id"],
            "name": m["name"],
            "total_budget": total_budget,
            "norm_weights": m["norm_weights"],
            "notes": m.get("notes", ""),
        }

        # Build concise activity list
        # Limit to 15 activities per call to avoid output truncation (8K token limit)
        MAX_ACTIVITIES_PER_CALL = 15
        activities_payload = []
        for idx, (_, row) in enumerate(city_df.iterrows()):
            if idx >= MAX_ACTIVITIES_PER_CALL:
                break
            biz_id = row["business_id"]
            activities_payload.append({
                "business_id": biz_id,
                "name": row["name"],
                "stars": float(row["stars"]),
                "review_count": int(row["review_count"]),
                "price_proxy": float(row["price_proxy"]),
                "value_score": per_user_scores[m["id"]].get(biz_id, 0.0),
                "tags5": {
                    d: float(row[f"tag_{d}"]) for d in DIMS
                },
            })

        # Prepare prompt
        system_instructions = """
You are a trip personalization model.
You receive a single user's profile and a list of candidate activities in one city.

For each activity, you must produce:
- adjusted_enjoyment_score (0–10, float)
- recommended_time_of_day ("morning" | "afternoon" | "evening" | "late night")
- note: MAX 15 words explaining why this user would or would not like it.

Consider:
- The user's normalized weights per dimension (nightlife, adventure, shopping, food, urban).
- The user's free-text notes.
- The activity's tag scores, rating, review_count, and price_proxy.
- The base value_score as a hint, but you can override it.

CRITICAL REQUIREMENTS:
1. Return ONLY raw JSON (no markdown code blocks, no ```json, no explanations, no extra text)
2. Keep "note" fields to MAX 15 words each
3. Return the complete JSON object - do not truncate
4. Start your response with { and end with }

Return valid JSON in this exact format:
{
  "user_id": "...",
  "city": "...",
  "results": [
    {
      "business_id": "...",
      "adjusted_enjoyment_score": 0.0,
      "recommended_time_of_day": "evening",
      "note": "Short 15-word max note here"
    }
  ]
}
"""

        user_payload = {
            "user_profile": user_profile,
            "city": city,
            "activities": activities_payload,
        }

        prompt = system_instructions.strip() + "\n\nUser + activities JSON:\n" + json.dumps(
            user_payload, indent=2
        )

        response = llm.invoke(prompt)
        
        # Extract JSON from response (handle markdown code blocks)
        content = extract_json_from_markdown(response.content)
        
        # Check for truncation (incomplete JSON)
        content_stripped = content.strip()
        if not content_stripped.endswith("}"):
            print(f"⚠️ WARNING: Response may be truncated for user {m['id']}")
            print(f"Response ends with: ...{content_stripped[-50:]}")
        
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            # Fallback: skip this user if parsing fails
            print(f"❌ JSON parsing failed for user {m['id']}: {str(e)}")
            print(f"Response preview: {response.content[:300]}")
            print(f"Extracted content preview: {content[:300]}")
            continue

        # Append per-activity results, but enforce schema
        for item in parsed.get("results", []):
            results.append({
                "user_id": parsed.get("user_id", m["id"]),
                "city": parsed.get("city", city),
                "business_id": item["business_id"],
                "adjusted_enjoyment_score": float(item["adjusted_enjoyment_score"]),
                "recommended_time_of_day": item["recommended_time_of_day"],
                "note": item["note"],
            })

    return results


def find_and_rerank_activities_tool(input_json: str) -> str:
    """
    Input JSON:
    {
      "city": "New Orleans",
      "group_members": [
        {
          "id": "u1",
          "name": "Alice",
          "total_budget": 900,
          "preference_weights": {...},
          "notes": "..."
        },
        ...
      ],
      "top_k": 10   # optional
    }

    Output JSON:
    {
      "city": "New Orleans",
      "activities": [
        {
          "user_id": "u1",
          "business_id": "abc123",
          "adjusted_enjoyment_score": 8.8,
          "recommended_time_of_day": "evening",
          "note": "Great for nightlife lovers..."
        },
        ...
      ]
    }
    """
    data = json.loads(input_json)
    city = data["city"]
    group_members = data["group_members"]
    top_k = data.get("top_k", 5)

    # Validate required fields in group_members
    required_fields = ["id", "name", "total_budget", "preference_weights"]
    for i, member in enumerate(group_members):
        missing = [field for field in required_fields if field not in member]
        if missing:
            return json.dumps({
                "error": f"Missing required fields for group_members[{i}]: {missing}. Each member must have: id, name, total_budget, preference_weights"
            })

    # 1) Compute base scores & pick top K activities
    base_info = compute_base_scores_for_city(city, group_members, top_k_activities=top_k)

    # 2) LLM rerank/enrich for each user-activity pair
    llm = build_llm_for_reranking()
    reranked = rerank_with_gemini(city, group_members, base_info, llm)

    # 3) Limit and optimize output to prevent overwhelming the agent
    # Group activities by user and keep only top 10 per user
    MAX_ACTIVITIES_PER_USER = 10
    activities_by_user = {}
    for activity in reranked:
        user_id = activity["user_id"]
        if user_id not in activities_by_user:
            activities_by_user[user_id] = []
        activities_by_user[user_id].append(activity)
    
    # Sort by score and keep top N per user
    limited_activities = []
    for user_id, user_activities in activities_by_user.items():
        sorted_activities = sorted(
            user_activities, 
            key=lambda x: x["adjusted_enjoyment_score"], 
            reverse=True
        )
        limited_activities.extend(sorted_activities[:MAX_ACTIVITIES_PER_USER])
    
    # Build compact output with summary
    # Calculate summary stats
    if limited_activities:
        scores = [a["adjusted_enjoyment_score"] for a in limited_activities]
        time_dist = {}
        for a in limited_activities:
            time = a["recommended_time_of_day"]
            time_dist[time] = time_dist.get(time, 0) + 1
    else:
        scores = []
        time_dist = {}
    
    output = {
        "city": city,
        "summary": {
            "total_activities": len(limited_activities),
            "score_range": [min(scores), max(scores)] if scores else [0, 0],
            "time_distribution": time_dist
        },
        "activities": limited_activities[:20],  # Hard limit: max 20 activities total
    }
    return json.dumps(output)  # No indent to reduce size


# ============================================================================
# SECTION 3: LANGCHAIN TOOL DEFINITIONS
# ============================================================================

@tool
def choose_city_tool(group_profile_json: str) -> str:
    """
    Choose the best city for the group based on preferences + budgets.
    Input: JSON with {group_members: [...]}
    Output: JSON with {"best_city": "...", "city_scores": [...], "explanation": "..."}
    """
    try:
        # Define constants (in case they're not in global scope)
        DIMS = ["nightlife", "adventure", "shopping", "food", "urban"]
        
        # Extract JSON from potentially wrapped text
        cleaned_json = extract_json_from_markdown(group_profile_json)
        group_profile = json.loads(cleaned_json)
        group_members = group_profile.get("group_members", [])
        
        if not group_members:
            return json.dumps({"error": "No group members provided"})
        
        # 1. Normalize preference weights for each member
        for m in group_members:
            m["norm_weights"] = normalize_weights(m["preference_weights"], DIMS)
        
        # 2. Compute group-level average preferences
        group_weights = {
            d: float(np.mean([m["norm_weights"][d] for m in group_members]))
            for d in DIMS
        }
        
        # 3. Load activities and compute city stats
        # Always load from CSV to avoid scope issues
        activities_df_local = pd.read_csv("activities.csv")
        
        # Compute city-level vibes & costs
        tag_cols = [f"tag_{d}" for d in DIMS]
        price_map = {1: 15, 2: 30, 3: 50, 4: 80}
        activities_df_local["price_proxy"] = activities_df_local["price_level"].map(price_map)
        
        city_stats = (
            activities_df_local
            .groupby("city")
            .agg(
                avg_tag_nightlife=("tag_nightlife", "mean"),
                avg_tag_adventure=("tag_adventure", "mean"),
                avg_tag_shopping=("tag_shopping", "mean"),
                avg_tag_food=("tag_food", "mean"),
                avg_tag_urban=("tag_urban", "mean"),
                avg_price_proxy=("price_proxy", "mean"),
                n_places=("business_id", "nunique"),
            )
            .reset_index()
        )
        
        # Filter cities with enough places
        city_stats = city_stats[city_stats["n_places"] >= 20].copy()
        
        # Build vibe vectors
        for d in DIMS:
            city_stats[f"vibe_{d}"] = city_stats[f"avg_tag_{d}"]
        
        # Normalize vibes per city
        vibe_cols = [f"vibe_{d}" for d in DIMS]
        city_stats["vibe_sum"] = city_stats[vibe_cols].sum(axis=1)
        city_stats[vibe_cols] = city_stats[vibe_cols].div(city_stats["vibe_sum"], axis=0)
        city_stats.drop(columns=["vibe_sum"], inplace=True)
        
        # Compute cost_index (0-1 scale)
        min_cost = city_stats["avg_price_proxy"].min()
        max_cost = city_stats["avg_price_proxy"].max()
        city_stats["cost_index"] = (city_stats["avg_price_proxy"] - min_cost) / (max_cost - min_cost + 1e-9)
        
        def cost_level(x):
            if x < 0.33:
                return "low"
            elif x < 0.66:
                return "medium"
            else:
                return "high"
        
        city_stats["typical_cost_level"] = city_stats["cost_index"].apply(cost_level)
        
        # 4. Score each city using your algorithm
        def score_city_row(row, group_weights, lambda_cost=0.6):
            city_vec = np.array([row[f"vibe_{d}"] for d in DIMS], dtype=float)
            w_vec = np.array([group_weights[d] for d in DIMS], dtype=float)
            
            vibe_score = float((city_vec * w_vec).sum())
            cost_index = float(row["cost_index"])
            
            # Penalize expensive cities
            cost_penalty = 1.0 / (1.0 + lambda_cost * cost_index)
            combined_score = vibe_score * cost_penalty
            
            return vibe_score, combined_score
        
        city_scores = []
        for _, row in city_stats.iterrows():
            vibe, combined = score_city_row(row, group_weights)
            city_scores.append({
                "city": row["city"],
                "vibe_score": vibe,
                "combined_score": combined,
                "avg_price_proxy": float(row["avg_price_proxy"]),
                "cost_index": float(row["cost_index"]),
                "typical_cost_level": row["typical_cost_level"],
                "n_places": int(row["n_places"]),
                "vibe_tags": {d: float(row[f"vibe_{d}"]) for d in DIMS},
            })
        
        # 5. Sort by combined_score and get best city
        city_scores_sorted = sorted(city_scores, key=lambda x: x["combined_score"], reverse=True)
        best_city = city_scores_sorted[0]
        
        # 6. Build result
        result = {
            "best_city": best_city["city"],
            "city_scores": city_scores_sorted[:10],  # Top 10 cities
            "explanation": f"{best_city['city']} has the highest combined score ({best_city['combined_score']:.3f}) based on group preferences and cost."
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


@tool
def search_flights_tool(
    origin: str,
    destination: str,
    depart_date: str,
    return_date: str,
    max_price: Optional[float] = None,
) -> str:
    """
    Search flights between origin and destination using SerpAPI.
    Takes city names and converts them to airport codes.
    Returns JSON list: [{price, airline, depart_time, arrive_time, legs}, ...]
    """
    import requests
    import os
    
    # City to airport code mapping for the 5 destination cities
    CITY_TO_AIRPORT = {
        "Philadelphia": "PHL",
        "Tucson": "TUS",
        "Tampa": "TPA",
        "Boise": "BOI",
        "New Orleans": "MSY",
    }
    
    # Common origin city airports (add more as needed)
    ORIGIN_AIRPORTS = {
        "Chicago": "ORD",  # O'Hare
        "New York": "JFK",
        "Los Angeles": "LAX",
        "San Francisco": "SFO",
        "Boston": "BOS",
        "Atlanta": "ATL",
        "Dallas": "DFW",
        "Seattle": "SEA",
    }
    
    try:
        # Convert city names to airport codes
        origin_code = ORIGIN_AIRPORTS.get(origin, origin.upper()[:3])  # Default to first 3 chars if not found
        dest_code = CITY_TO_AIRPORT.get(destination, destination.upper()[:3])
        
        # If already airport codes, use them directly
        if len(origin) == 3 and origin.isupper():
            origin_code = origin
        if len(destination) == 3 and destination.isupper():
            dest_code = destination
        
        # Normalize to uppercase
        origin_code = origin_code.strip().upper()
        dest_code = dest_code.strip().upper()
        
        SERPAPI_KEY = os.getenv('SERPAPI_KEY')
        
        if not SERPAPI_KEY:
            return json.dumps({"error": "SERPAPI_KEY not found in environment variables. Please set SERPAPI_KEY."})
        
        # API call
        params = {
            "engine": "google_flights",
            "departure_id": origin_code,
            "arrival_id": dest_code,
            "outbound_date": depart_date,
            "return_date": return_date,
            "currency": "USD",
            "hl": "en",
            "api_key": SERPAPI_KEY
        }
        
        response = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        
        if response.status_code != 200:
            return json.dumps({"error": f"API Error: {response.status_code}"})
        
        data = response.json()
        
        if "error" in data:
            return json.dumps({"error": data["error"]})
        
        # Get all flights
        all_flights = data.get("best_flights", []) + data.get("other_flights", [])
        
        if not all_flights:
            return json.dumps({"error": "No flights found"})
        
        # Sort by price, get top 3
        cheapest = sorted(all_flights, key=lambda x: x.get('price', 999999))[:3]
        
        # Format results to match expected format
        results = []
        for flight in cheapest:
            legs = flight.get("flights", [])
            if not legs:
                continue
            
            # Extract times
            depart_time = legs[0].get("departure_airport", {}).get("time", "")
            arrive_time = legs[-1].get("arrival_airport", {}).get("time", "")
            
            # Format times to match expected format (YYYY-MM-DDTHH:MM)
            if depart_time and "T" not in depart_time:
                depart_time = f"{depart_date}T{depart_time}"
            if arrive_time and "T" not in arrive_time:
                arrive_time = f"{return_date}T{arrive_time}"
            
            results.append({
                "price": flight.get("price"),
                "airline": legs[0].get("airline", "Unknown"),
                "depart_time": depart_time,
                "arrive_time": arrive_time,
                "legs": len(legs),
            })
        
        # Filter by max_price if provided
        if max_price is not None:
            results = [f for f in results if f.get("price", 999999) <= max_price]
        
        return json.dumps(results)
    
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def search_hotels_tool(
    city: str,
    checkin: str,
    checkout: str,
    max_price: Optional[float] = None,
    min_rating: float = 0.0,
) -> str:
    """
    Minimal Google Hotels search via SerpAPI.
    
    Inputs:
      - city: e.g. "New Orleans"
      - checkin: "YYYY-MM-DD"
      - checkout: "YYYY-MM-DD"
      - max_price: optional upper bound in USD
      - min_rating: minimum rating
    
    Returns:
      JSON list of up to 5 hotels (filtered by max_price and min_rating):
        [
          {
            "name": "Hotel Name",
            "rating": 4.5,
            "area": "Downtown",
            "price_per_night": 120,
          },
          ...
        ]
    """
    import os
    import requests

    try:
        SERPAPI_KEY = os.getenv("SERPAPI_KEY")
        if not SERPAPI_KEY:
            return json.dumps({
                "error": "SERPAPI_KEY not found in environment variables. Please set SERPAPI_KEY."
            })

        params = {
            "engine": "google_hotels",
            "q": city,
            "check_in_date": checkin,
            "check_out_date": checkout,
            "adults": 2,
            "currency": "USD",
            "hl": "en",
            "gl": "us",
            "api_key": SERPAPI_KEY,
        }

        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        if resp.status_code != 200:
            return json.dumps({"error": f"API Error: {resp.status_code}", "details": resp.text[:500]})

        data = resp.json()

        # Hotels are often under "properties" or "results" / "hotel_results"
        properties = data.get("properties") or data.get("hotel_results") or data.get("results") or []

        if not properties:
            return json.dumps({
                "error": "No hotels found in response under properties/hotel_results/results",
                "top_level_keys": list(data.keys()),
            }, indent=2)

        hotels = []
        for p in properties:
            name = p.get("name") or p.get("title") or "Unknown"

            # Rating - according to API docs, it's "overall_rating"
            rating = p.get("overall_rating") or p.get("rating")

            # Location/Area - according to API docs, location is an object
            area = "Unknown"
            location = p.get("location")
            if location:
                if isinstance(location, dict):
                    area = location.get("neighborhood") or location.get("city") or location.get("address") or "Unknown"
                elif isinstance(location, str):
                    area = location
            else:
                # Fallback to direct fields
                area = p.get("neighborhood") or p.get("address") or "Unknown"

            # Price extraction - according to API docs:
            # rate_per_night.extracted_lowest or total_rate.extracted_lowest
            price_per_night = None
            
            # Try rate_per_night.extracted_lowest first (per API docs)
            rate_per_night = p.get("rate_per_night")
            if rate_per_night and isinstance(rate_per_night, dict):
                price_per_night = rate_per_night.get("extracted_lowest")
            
            # Fallback to total_rate.extracted_lowest
            if price_per_night is None:
                total_rate = p.get("total_rate")
                if total_rate and isinstance(total_rate, dict):
                    price_per_night = total_rate.get("extracted_lowest")
            
            # Additional fallbacks for compatibility
            if price_per_night is None:
                raw_price = p.get("rate") or p.get("price")
                if isinstance(raw_price, (int, float)):
                    price_per_night = float(raw_price)
                elif isinstance(raw_price, dict):
                    price_per_night = raw_price.get("extracted_lowest") or raw_price.get("rate") or raw_price.get("price")
                    if price_per_night:
                        try:
                            price_per_night = float(price_per_night)
                        except (ValueError, TypeError):
                            price_per_night = None
                elif isinstance(raw_price, str):
                    # Extract number from string like "$120/night" or "120"
                    numbers = re.findall(r'\d+\.?\d*', raw_price.replace(',', ''))
                    if numbers:
                        try:
                            price_per_night = float(numbers[0])
                        except (ValueError, TypeError):
                            price_per_night = None
            
            # Skip hotels without valid price or rating
            if price_per_night is None:
                continue
            
            # Convert rating to float for filtering
            try:
                rating_float = float(rating) if rating is not None else 0.0
            except (ValueError, TypeError):
                rating_float = 0.0
            
            # Apply min_rating filter
            if rating_float < min_rating:
                continue
            
            # Apply max_price filter
            if max_price is not None and price_per_night > max_price:
                continue

            hotels.append({
                "name": name,
                "rating": rating_float,
                "area": area or "Unknown",
                "price_per_night": price_per_night,
            })

        # Return top 5 hotels
        return json.dumps(hotels[:5], indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def score_activities_tool(input_json: str) -> str:
    """
    Score and rank activities for the chosen city.
    Input: JSON {city, group_members, top_k}
    Output: JSON {city, activities: [{user_id, business_id, adjusted_enjoyment_score, ...}]}
    """
    try:
        # Extract JSON from potentially wrapped text
        cleaned_json = extract_json_from_markdown(input_json)
        return find_and_rerank_activities_tool(cleaned_json)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def fairness_tool(group_costs_json: str) -> str:
    """
    Compute fairness score from group costs.
    Input: JSON {"group_members": [{name, total_budget, flight_price, hotel_share, activities_spend}, ...]}
    Output: JSON {trip_fairness_score, mean_ratio, std_ratio, per_person: [...]}
    """
    try:
        data = json.loads(group_costs_json)
        summary = compute_fairness_summary(data["group_members"])
        return json.dumps(summary)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# SECTION 4: LLM & AGENT SETUP
# ============================================================================

def setup_agent():
    """Set up the LLM and create the agent"""
    # Get API key from environment variable (never hardcode it!)
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not set. Please set it as an environment variable.")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )

    # Create tools list
    tools = [
        choose_city_tool,
        search_flights_tool,
        search_hotels_tool,
        score_activities_tool,
        fairness_tool,
    ]

    # The working prompt
    working_prompt = """You are FairTripAgent. Plan a 2-day group trip.

WORKFLOW:
1. Call choose_city_tool with group_members JSON
2. EXPLAIN what city was chosen and why (1-2 sentences)
3. Call search_flights_tool for both trip windows
4. EXPLAIN what flights you found and which is best (1-2 sentences)
5. Call search_hotels_tool for that window
6. EXPLAIN what hotels you found and which you'll choose (1-2 sentences)
7. Call score_activities_tool for the destination
8. EXPLAIN what activities you found - use summary if available (1-2 sentences)
9. Build a 2-day itinerary (3 activities/day: morning, afternoon, evening)
   - For EACH activity, include reasoning that references the user's "notes" field
   - Explain WHY this activity would interest each user based on their preferences and notes
   - Format: {
       "day1": {
         "morning": {
           "activity": "Activity name/description",
           "reasoning": "Why this interests the group, referencing specific user notes (e.g., 'Alice loves walkable neighborhoods, so this urban walking tour fits her preference for speakeasies and walkable areas')"
         },
         "afternoon": {...},
         "evening": {...}
       },
       "day2": {...}
     }
   - IMPORTANT: The reasoning MUST reference specific user names and quote or paraphrase their "notes" field
   - Example reasoning: "This speakeasy tour appeals to Alice who 'loves walkable neighborhoods and speakeasies' (from her notes), and Emma who 'enjoys nightlife' and is 'happy to splurge a bit on nightlife' (from her notes). The urban walking aspect fits Alice's preference, while the evening timing matches Emma's nightlife interests."
10. Call fairness_tool with costs
11. EXPLAIN the fairness results (1-2 sentences)
12. Return final JSON with: chosen_city, chosen_trip_window, chosen_flight, chosen_hotel, itinerary, fairness_summary, explanation

CRITICAL: You MUST write a text explanation after EVERY tool call. Never skip this step. If a tool returns data, summarize it in 1-2 sentences before proceeding."""

    # Create the agent
    print("✅ Creating agent...")
    agent = create_react_agent(
        llm,
        tools,
        prompt=working_prompt
    )
    print("✅ Agent created successfully!")
    
    return agent


# ============================================================================
# SECTION 5: EXECUTE AGENT
# ============================================================================

def extract_message_content(msg):
    """Safely extract content from a message, handling both string and list formats."""
    if not msg.content:
        return ""
    
    if isinstance(msg.content, list):
        # Handle list of items - could be strings or dicts with 'text' key
        parts = []
        for item in msg.content:
            if isinstance(item, dict):
                # Extract text from dict items (e.g., {'type': 'text', 'text': '...'})
                if 'text' in item:
                    parts.append(str(item['text']))
                else:
                    # Fallback: convert entire dict to string
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return " ".join(parts)
    else:
        return str(msg.content)


def build_trip_request(group_members: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Construct the payload the agent expects from an arbitrary list of group members.
    """
    most_common_window = find_most_common_window(group_members)
    if not most_common_window:
        raise ValueError("Unable to determine a common trip window from the provided members.")

    all_windows = []
    seen_windows = set()
    for member in group_members:
        for window in member.get("preferred_windows", []):
            trip_start = window.get("trip_start")
            trip_end = window.get("trip_end")
            if not trip_start or not trip_end:
                continue
            window_key = (trip_start, trip_end)
            if window_key in seen_windows:
                continue
            seen_windows.add(window_key)
            all_windows.append({
                "window_id": f"W{len(all_windows) + 1}",
                "trip_start": trip_start,
                "trip_end": trip_end,
            })

    if not all_windows:
        raise ValueError("No valid trip windows provided by the group members.")

    return {
        "most_common_window": most_common_window,
        "trip_windows": all_windows[:3],
        "group_members": group_members,
    }


def parse_agent_final_message(final_message: str) -> Dict[str, Any]:
    """
    Try to convert the agent's final reply into structured JSON.
    Falls back to returning the raw text if parsing fails.
    """
    if not final_message:
        return {}

    extracted = extract_json_from_markdown(final_message)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return {"raw_text": final_message}


def extract_itinerary_from_tool_messages(messages: List[Any]) -> Dict[str, Any]:
    """
    Extract itinerary information from agent tool messages when final message is empty.
    Reconstructs the structured response from tool outputs.
    """
    result = {
        "chosen_city": None,
        "chosen_trip_window": None,
        "chosen_flight": {},
        "chosen_hotel": None,
        "itinerary": {},
        "fairness_summary": None,
        "explanation": "Itinerary reconstructed from agent tool outputs."
    }
    
    # Load activities CSV to get activity names
    activities_df = None
    try:
        if os.path.exists("activities.csv"):
            activities_df = pd.read_csv("activities.csv")
    except:
        pass
    
    # Extract data from tool messages
    for msg in messages:
        # Check if it's a ToolMessage
        if hasattr(msg, 'name') and hasattr(msg, 'content'):
            tool_name = msg.name
            tool_content = msg.content
            
            # Parse city selection
            if tool_name == 'choose_city_tool':
                try:
                    city_data = json.loads(tool_content)
                    if isinstance(city_data, dict):
                        result["chosen_city"] = city_data.get("best_city")
                        if city_data.get("explanation"):
                            result["explanation"] = city_data.get("explanation")
                except:
                    pass
            
            # Parse flight search results
            elif tool_name == 'search_flights_tool':
                try:
                    flight_data = json.loads(tool_content)
                    if isinstance(flight_data, dict) and "error" not in flight_data:
                        # Flight found - would need to extract from tool call context
                        pass
                    elif isinstance(flight_data, dict) and "error" in flight_data:
                        # No flights found - this is expected for some destinations
                        result["chosen_flight"] = {"error": "No flights found for this route"}
                except:
                    pass
            
            # Parse hotel search results
            elif tool_name == 'search_hotels_tool':
                try:
                    hotels = json.loads(tool_content)
                    if isinstance(hotels, list) and len(hotels) > 0:
                        # Select the best hotel (highest rating, reasonable price)
                        best_hotel = max(hotels, key=lambda h: (h.get('rating', 0), -h.get('price_per_night', 9999)))
                        result["chosen_hotel"] = best_hotel
                except:
                    pass
            
            # Parse activity scoring results
            elif tool_name == 'score_activities_tool':
                try:
                    activities_data = json.loads(tool_content)
                    if isinstance(activities_data, dict):
                        activities = activities_data.get("activities", [])
                        city = activities_data.get("city")
                        if city:
                            result["chosen_city"] = city
                        
                        # Build itinerary from activities
                        # Group by time of day and get top activities
                        morning_activities = [a for a in activities if a.get("recommended_time_of_day") == "morning"]
                        afternoon_activities = [a for a in activities if a.get("recommended_time_of_day") == "afternoon"]
                        evening_activities = [a for a in activities if a.get("recommended_time_of_day") == "evening"]
                        
                        def get_activity_name(business_id):
                            """Get activity name from CSV if available"""
                            if activities_df is not None:
                                try:
                                    activity_row = activities_df[activities_df['business_id'] == business_id]
                                    if not activity_row.empty:
                                        return activity_row.iloc[0].get('name', 'Activity')
                                except:
                                    pass
                            return "Activity"
                        
                        # Day 1 itinerary
                        day1 = {}
                        
                        if morning_activities:
                            top_morning = max(morning_activities, key=lambda a: a.get("adjusted_enjoyment_score", 0))
                            activity_name = get_activity_name(top_morning.get("business_id", ""))
                            day1["morning"] = {
                                "activity": activity_name,
                                "reasoning": top_morning.get("note", "Selected based on group preferences")
                            }
                        
                        if afternoon_activities:
                            top_afternoon = max(afternoon_activities, key=lambda a: a.get("adjusted_enjoyment_score", 0))
                            activity_name = get_activity_name(top_afternoon.get("business_id", ""))
                            day1["afternoon"] = {
                                "activity": activity_name,
                                "reasoning": top_afternoon.get("note", "Selected based on group preferences")
                            }
                        
                        if evening_activities:
                            top_evening = max(evening_activities, key=lambda a: a.get("adjusted_enjoyment_score", 0))
                            activity_name = get_activity_name(top_evening.get("business_id", ""))
                            day1["evening"] = {
                                "activity": activity_name,
                                "reasoning": top_evening.get("note", "Selected based on group preferences")
                            }
                        
                        if day1:
                            result["itinerary"]["day1"] = day1
                            
                        # For day 2, use next best activities
                        day2 = {}
                        used_business_ids = set()
                        if day1.get("morning"):
                            used_business_ids.add(morning_activities[0].get("business_id") if morning_activities else "")
                        if day1.get("afternoon"):
                            used_business_ids.add(afternoon_activities[0].get("business_id") if afternoon_activities else "")
                        if day1.get("evening"):
                            used_business_ids.add(evening_activities[0].get("business_id") if evening_activities else "")
                        
                        # Get second-best activities for day 2
                        morning_remaining = [a for a in morning_activities if a.get("business_id") not in used_business_ids]
                        afternoon_remaining = [a for a in afternoon_activities if a.get("business_id") not in used_business_ids]
                        evening_remaining = [a for a in evening_activities if a.get("business_id") not in used_business_ids]
                        
                        if morning_remaining:
                            top_morning2 = max(morning_remaining, key=lambda a: a.get("adjusted_enjoyment_score", 0))
                            activity_name = get_activity_name(top_morning2.get("business_id", ""))
                            day2["morning"] = {
                                "activity": activity_name,
                                "reasoning": top_morning2.get("note", "Selected based on group preferences")
                            }
                        
                        if afternoon_remaining:
                            top_afternoon2 = max(afternoon_remaining, key=lambda a: a.get("adjusted_enjoyment_score", 0))
                            activity_name = get_activity_name(top_afternoon2.get("business_id", ""))
                            day2["afternoon"] = {
                                "activity": activity_name,
                                "reasoning": top_afternoon2.get("note", "Selected based on group preferences")
                            }
                        
                        if evening_remaining:
                            top_evening2 = max(evening_remaining, key=lambda a: a.get("adjusted_enjoyment_score", 0))
                            activity_name = get_activity_name(top_evening2.get("business_id", ""))
                            day2["evening"] = {
                                "activity": activity_name,
                                "reasoning": top_evening2.get("note", "Selected based on group preferences")
                            }
                        
                        if day2:
                            result["itinerary"]["day2"] = day2
                            
                except Exception as e:
                    import traceback
                    # Silently continue - reconstruction is best effort
                    pass
    
    return result


def run_trip_planner_for_group(group_members: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Entry point used by external callers (e.g., Streamlit) to execute the agent workflow.
    """
    if not os.path.exists("activities.csv"):
        raise FileNotFoundError("activities.csv not found. Place it in the project root.")

    agent = setup_agent()
    trip_request = build_trip_request(group_members)
    test_msg = f"Plan a trip for this group: {json.dumps(trip_request)}"

    result = agent.invoke({"messages": [("user", test_msg)]})
    final_message = extract_message_content(result["messages"][-1])
    structured_response = parse_agent_final_message(final_message)
    
    # If final message is empty or parsing failed, try to reconstruct from tool messages
    if not structured_response or structured_response == {} or (len(structured_response) == 1 and "raw_text" in structured_response):
        # Extract trip window from request
        most_common_window = trip_request.get("most_common_window", {})
        trip_window = {
            "trip_start": most_common_window.get("trip_start"),
            "trip_end": most_common_window.get("trip_end")
        }
        
        # Reconstruct from tool messages
        reconstructed = extract_itinerary_from_tool_messages(result["messages"])
        reconstructed["chosen_trip_window"] = trip_window
        
        # If we got a city, use the reconstructed response
        if reconstructed.get("chosen_city"):
            structured_response = reconstructed

    return {
        "agent_result": result,
        "trip_request": trip_request,
        "final_message": final_message,
        "structured_response": structured_response,
    }


def main():
    """Main execution function"""
    print("="*80)
    print("AI TRIP PLANNER - Starting Execution")
    print("="*80)
    
    # Check for required files
    if not os.path.exists("activities.csv"):
        print("❌ ERROR: activities.csv not found in current directory")
        print("   Please ensure activities.csv is in the same directory as this script")
        return
    
    # Check for SERPAPI_KEY (GOOGLE_API_KEY is hardcoded in setup_agent())
    if not os.getenv("SERPAPI_KEY"):
        print("⚠️  WARNING: SERPAPI_KEY not set")
        print("   Flight and hotel searches will fail without this key")
        print("   Set it with: export SERPAPI_KEY='your_key'")
    
    try:
        run_result = run_trip_planner_for_group(group_members)
    except Exception as exc:
        print(f"❌ ERROR: {exc}")
        return

    most_common_window = run_result["trip_request"]["most_common_window"]
    print(f"\n📅 Most common window: {most_common_window['trip_start']} to {most_common_window['trip_end']}")
    print(f"   Preferred by {most_common_window['member_count']} out of {most_common_window['total_members']} members")
    print(f"\n✈️  Origin cities:")
    for member in group_members:
        print(f"   {member['name']}: {member['origin_city']}")

    print("\n" + "="*80)
    print("🚀 EXECUTING AGENT")
    print("="*80 + "\n")

    result = run_result["agent_result"]

    # Display results
    print(f"\n📨 Total messages: {len(result['messages'])}\n")
    
    for i, msg in enumerate(result['messages']):
        print(f"{i+1}. [{type(msg).__name__}] ", end="")
        
        # Safely extract content (handles both string and list)
        content_str = extract_message_content(msg)
        
        if content_str:
            # Print full message content
            print(content_str)
        else:
            print("(empty)")
    
    print("\n" + "="*80)
    print("📝 FINAL RESPONSE:")
    print("="*80)
    
    final_content_str = run_result["final_message"]
    print(final_content_str)
    
    print("\n" + "="*80)
    print("✅ Execution complete!")
    print("="*80)


if __name__ == "__main__":
    main()

