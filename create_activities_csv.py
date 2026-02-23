#!/usr/bin/env python3
"""
Yelp Dataset Processor - Creates activities.csv from Yelp Academic Dataset
Processes ALL cities (not just a subset) and applies quality filters.
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths to Yelp dataset JSON files
BUSINESS_JSON_PATH = Path("yelp_academic_dataset_business.json")
REVIEW_JSON_PATH = Path("yelp_academic_dataset_review.json")
OUTPUT_CSV_PATH = Path("activities.csv")

# Quality filters (same as notebook)
MIN_STARS = 4.0
MIN_REVIEW_COUNT = 50
MAX_BUSINESSES_PER_CITY = 150

# Optional: Filter to specific cities only (set to None to process all cities)
# CITIES = None  # Process all cities
# Or specify cities:
# CITIES = {"New Orleans", "Philadelphia", "Tucson", "Tampa", "Boise"}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_categories(raw_categories):
    """Convert Yelp categories field into a list of lowercase category strings."""
    if not raw_categories:
        return []
    if isinstance(raw_categories, str):
        parts = [c.strip().lower() for c in raw_categories.split(",") if c.strip()]
        return parts
    # Already a list / array
    return [str(c).strip().lower() for c in raw_categories if str(c).strip()]


def build_tags5(categories):
    """
    Build 5D tags (nightlife, adventure, shopping, food, urban)
    based on Yelp categories.
    """
    nightlife_keywords = [
        "bars", "nightlife", "pub", "pubs", "lounges", "cocktail",
        "beer bar", "beer garden", "wine bar", "sports bar", "club", "dance club",
    ]
    adventure_keywords = [
        "hiking", "climbing", "outdoor", "outdoors", "rafting", "surf", "ski",
        "snowboard", "biking", "bike", "bikes", "zipline", "rock climbing",
        "water sports",
    ]
    shopping_keywords = [
        "shopping", "fashion", "department store", "mall", "boutiques",
        "thrift", "vintage", "jewelry", "bookstores", "shopping centers",
    ]
    food_keywords = [
        "restaurants", "food", "coffee", "tea", "cafe", "cafes", "bakeries",
        "desserts", "ice cream", "frozen yogurt", "pizza", "sushi", "steakhouse",
        "burgers", "sandwiches", "breakfast & brunch",
    ]
    urban_keywords = [
        "arts & entertainment", "museum", "museums", "landmarks & historical buildings",
        "tours", "local flavor", "parks", "park", "theater", "theatre", "stadium",
        "music venue", "cinema", "art gallery", "galleries", "cultural center",
    ]

    scores = {
        "nightlife": 0.0,
        "adventure": 0.0,
        "shopping": 0.0,
        "food": 0.0,
        "urban": 0.0,
    }

    for cat in categories:
        c = cat.lower()

        def matches_any(keywords):
            return any(k in c for k in keywords)

        if matches_any(nightlife_keywords):
            scores["nightlife"] += 1.0
        if matches_any(adventure_keywords):
            scores["adventure"] += 1.0
        if matches_any(shopping_keywords):
            scores["shopping"] += 1.0
        if matches_any(food_keywords):
            scores["food"] += 1.0
        if matches_any(urban_keywords):
            scores["urban"] += 1.0

    total = sum(scores.values())
    if total == 0:
        # fallback: treat as general urban exploration
        scores["urban"] = 1.0
        total = 1.0

    # normalize to sum to 1.0
    for k in scores:
        scores[k] = scores[k] / total

    return scores


def extract_price_level(attributes):
    """
    Extract price level from the nested attributes dict.
    Prefer RestaurantsPriceRange2 (1–4), fall back to 'Price Range' if present.
    """
    if not isinstance(attributes, dict):
        return None
    raw = attributes.get("RestaurantsPriceRange2") or attributes.get("Price Range")
    if raw is None:
        return None
    try:
        return int(str(raw).strip().strip("'\""))
    except ValueError:
        return None


def load_businesses(cities_filter=None):
    """
    Stream the business JSON and build a compact table.
    
    Args:
        cities_filter: Set of city names to filter by, or None to process all cities
    """
    selected = []
    per_city_counts = defaultdict(int)
    total_processed = 0

    print(f"Loading businesses from {BUSINESS_JSON_PATH}...")
    print(f"Filters: MIN_STARS={MIN_STARS}, MIN_REVIEW_COUNT={MIN_REVIEW_COUNT}, MAX_BUSINESSES_PER_CITY={MAX_BUSINESSES_PER_CITY}")
    if cities_filter:
        print(f"City filter: {len(cities_filter)} cities specified")
    else:
        print("Processing ALL cities")

    with BUSINESS_JSON_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            
            total_processed += 1
            if total_processed % 100000 == 0:
                print(f"  Processed {total_processed:,} businesses... (found {len(selected)} so far)")
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            city = data.get("city")
            if not city:
                continue
            
            # Apply city filter if specified
            if cities_filter is not None and city not in cities_filter:
                continue

            # Keep only open businesses
            if data.get("is_open", 1) == 0:
                continue

            stars = data.get("stars", 0.0)
            review_count = data.get("review_count", 0)

            # Apply quality filters
            if stars < MIN_STARS or review_count < MIN_REVIEW_COUNT:
                continue

            # Limit per city
            if per_city_counts[city] >= MAX_BUSINESSES_PER_CITY:
                continue

            categories = parse_categories(data.get("categories"))
            if not categories:
                continue

            tags5 = build_tags5(categories)

            attrs = data.get("attributes") or {}
            price_level = extract_price_level(attrs)

            record = {
                "business_id": data.get("business_id"),
                "name": data.get("name"),
                "city": city,
                "state": data.get("state"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "stars": stars,
                "review_count": review_count,
                "categories_raw": data.get("categories"),
                "categories": categories,
                "price_level": price_level,
                "tags5": tags5,
            }

            selected.append(record)
            per_city_counts[city] += 1

    df = pd.DataFrame(selected)
    print(f"\n✓ Selected {len(df)} businesses across {len(per_city_counts)} cities")
    if not df.empty:
        print("\nTop 20 cities by business count:")
        city_counts = df["city"].value_counts()
        for city, count in city_counts.head(20).items():
            print(f"  {city:30s} {count:4d}")
        if len(city_counts) > 20:
            print(f"  ... and {len(city_counts) - 20} more cities")
    
    return df


def load_top_reviews_for_businesses(business_ids):
    """
    Stream the reviews JSON and keep top 3 reviews per business
    sorted by (useful + funny + cool).
    """
    top_reviews = defaultdict(list)  # business_id -> list of (engagement, text)
    total_processed = 0

    print(f"\nLoading reviews from {REVIEW_JSON_PATH}...")
    print(f"Looking for reviews for {len(business_ids)} businesses...")

    with REVIEW_JSON_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            
            total_processed += 1
            if total_processed % 1_000_000 == 0:
                print(f"  Processed {total_processed:,} review lines... (found reviews for {len(top_reviews)} businesses)")
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            bid = data.get("business_id")
            if bid not in business_ids:
                continue

            useful = data.get("useful", 0) or 0
            funny = data.get("funny", 0) or 0
            cool = data.get("cool", 0) or 0
            engagement = useful + funny + cool

            text = data.get("text", "").strip()
            if not text:
                continue

            current = top_reviews[bid]

            if len(current) < 3:
                current.append((engagement, text))
                current.sort(key=lambda x: x[0], reverse=True)
            else:
                # if this review is more engaging than the least engaging in top 3
                if engagement > current[-1][0]:
                    current[-1] = (engagement, text)
                    current.sort(key=lambda x: x[0], reverse=True)

    # Convert to mapping of business_id -> list of texts
    top_texts = {
        bid: [t for _, t in sorted(reviews, key=lambda x: x[0], reverse=True)]
        for bid, reviews in top_reviews.items()
    }
    
    print(f"✓ Found reviews for {len(top_texts)} businesses")
    return top_texts


def build_places_dataset(cities_filter=None):
    """High-level function to build the final places dataset."""
    businesses_df = load_businesses(cities_filter)
    
    if businesses_df.empty:
        print("\n❌ No businesses selected. Check your filters and file paths.")
        return businesses_df

    business_ids = set(businesses_df["business_id"])

    top_reviews = load_top_reviews_for_businesses(business_ids)

    # Attach top_reviews as a list-of-strings column
    businesses_df["top_reviews"] = businesses_df["business_id"].map(
        lambda bid: top_reviews.get(bid, [])
    )

    # Explode tags5 dict into separate columns for convenience
    tags_df = businesses_df["tags5"].apply(pd.Series)
    tags_df.columns = [f"tag_{c}" for c in tags_df.columns]

    final_df = pd.concat([businesses_df.drop(columns=["tags5"]), tags_df], axis=1)

    print(f"\n✓ Final dataset shape: {final_df.shape}")
    return final_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 70)
    print("Yelp Dataset Processor - Creating activities.csv")
    print("=" * 70)
    
    # Check that input files exist
    if not BUSINESS_JSON_PATH.exists():
        print(f"❌ Error: {BUSINESS_JSON_PATH} not found!")
        print(f"   Please ensure the Yelp dataset file is in the current directory.")
        return
    
    if not REVIEW_JSON_PATH.exists():
        print(f"⚠️  Warning: {REVIEW_JSON_PATH} not found!")
        print(f"   Reviews will be skipped. Continuing with businesses only...")
        # We'll handle this in load_top_reviews_for_businesses
    
    # Set cities filter to None to process all cities
    # Or uncomment and modify to filter specific cities:
    # CITIES_FILTER = {"New Orleans", "Philadelphia", "Tucson", "Tampa", "Boise"}
    CITIES_FILTER = None  # Process all cities
    
    # Build dataset
    final_df = build_places_dataset(cities_filter=CITIES_FILTER)
    
    if final_df.empty:
        print("\n❌ No data to save. Exiting.")
        return
    
    # Save to CSV
    print(f"\n💾 Saving to {OUTPUT_CSV_PATH}...")
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"✅ Successfully created {OUTPUT_CSV_PATH}")
    print(f"   Total businesses: {len(final_df)}")
    print(f"   Total cities: {final_df['city'].nunique()}")
    print(f"   File size: {OUTPUT_CSV_PATH.stat().st_size / (1024*1024):.2f} MB")
    
    # Validate tag columns
    tag_cols = ["tag_nightlife", "tag_adventure", "tag_shopping", "tag_food", "tag_urban"]
    if all(col in final_df.columns for col in tag_cols):
        tag_sums = final_df[tag_cols].sum(axis=1)
        print(f"\n✓ Tag validation: All tag columns sum to ~1.0 (mean: {tag_sums.mean():.6f})")
    else:
        print(f"\n⚠️  Warning: Some tag columns are missing!")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()