#!/usr/bin/env python3
"""
Streamlit UI for the AI Trip Planner.
Enhanced UI with dynamic traveler count and beautiful itinerary display.
"""

import json
from datetime import date, timedelta
from typing import List, Dict, Any

import streamlit as st

from trip_planner import run_trip_planner_for_group

# Page configuration
st.set_page_config(
    page_title="AI Group Trip Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .itinerary-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .activity-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">✈️ AI Group Trip Planner</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plan the perfect group getaway with AI-powered itinerary generation</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    num_travelers = st.number_input(
        "Number of Travelers",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="How many people are going on the trip?"
    )
    num_windows = st.number_input(
        "Number of Date Windows",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="How many different date options should each traveler provide?"
    )
    st.markdown("---")
    st.markdown("**Requirements:**")
    st.markdown("- `GOOGLE_API_KEY` environment variable")
    st.markdown("- `SERPAPI_KEY` environment variable")
    st.markdown("- `activities.csv` file in project directory")
    
    st.markdown("---")
    st.markdown("**💡 Tip:**")
    st.markdown("Fill out all traveler details and click 'Generate Itinerary' to create your perfect group trip!")


def default_dates(offset_weeks: int) -> Dict[str, date]:
    """Helper that produces a weekend window offset from today."""
    today = date.today()
    start = today + timedelta(weeks=offset_weeks)
    end = start + timedelta(days=1)
    return {"start": start, "end": end}


def build_member_payload(form_state: Dict[str, Any], idx: int, num_windows: int) -> Dict[str, Any]:
    """Transform Streamlit inputs into the structure the planner expects."""
    windows: List[Dict[str, str]] = []
    for window_idx in range(1, num_windows + 1):
        start_key = f"trip_start_{idx}_{window_idx}"
        end_key = f"trip_end_{idx}_{window_idx}"
        start_val = form_state.get(start_key)
        end_val = form_state.get(end_key)
        if start_val and end_val:
            windows.append(
                {
                    "trip_start": start_val.isoformat(),
                    "trip_end": end_val.isoformat(),
                }
            )

    if not windows:
        raise ValueError(f"Traveler {idx + 1} must specify at least one preferred window.")

    preferences = {
        dim: form_state[f"{dim}_{idx}"]
        for dim in ["nightlife", "adventure", "shopping", "food", "urban"]
    }

    return {
        "id": f"user_{idx+1}",
        "name": form_state[f"name_{idx}"].strip(),
        "origin_city": form_state[f"origin_{idx}"].strip().upper(),
        "total_budget": form_state[f"budget_{idx}"],
        "preference_weights": preferences,
        "notes": form_state.get(f"notes_{idx}", "").strip(),
        "preferred_windows": windows,
    }


# Main form
with st.form("traveler_form"):
    st.header("👥 Traveler Information")
    st.markdown("Fill out the details for each traveler. Use the sliders to indicate preferences (0 = dislike, 5 = love).")
    
    form_state: Dict[str, Any] = {}
    
    # Create tabs for better organization if many travelers
    if num_travelers <= 5:
        tabs = st.tabs([f"Traveler {i+1}" for i in range(num_travelers)])
    else:
        st.info(f"Showing {num_travelers} travelers. Scroll down to see all.")
        tabs = [st.container() for _ in range(num_travelers)]
    
    for idx in range(num_travelers):
        with tabs[idx]:
            defaults = default_dates(idx)
            
            col1, col2 = st.columns(2)
            with col1:
                form_state[f"name_{idx}"] = st.text_input(
                    "Name",
                    key=f"name_input_{idx}",
                    placeholder="Enter traveler name"
                )
            with col2:
                form_state[f"origin_{idx}"] = st.text_input(
                    "Origin Airport Code",
                    key=f"origin_input_{idx}",
                    placeholder="e.g., ORD, JFK, LAX",
                    help="3-letter airport code"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                form_state[f"budget_{idx}"] = st.number_input(
                    "Total Budget (USD)",
                    min_value=500,
                    max_value=20000,
                    value=5000,
                    step=100,
                    key=f"budget_input_{idx}"
                )
            with col2:
                st.write("")  # Spacer for alignment
            
            form_state[f"notes_{idx}"] = st.text_area(
                "Personal Notes / Preferences",
                key=f"notes_input_{idx}",
                placeholder="E.g., Loves nightlife, hates early mornings, vegetarian, prefers walkable neighborhoods...",
                height=100
            )

            st.markdown("**Preference Weights**")
            pref_cols = st.columns(5)
            for i, dim in enumerate(["nightlife", "adventure", "shopping", "food", "urban"]):
                with pref_cols[i]:
                    form_state[f"{dim}_{idx}"] = st.slider(
                        dim.capitalize(),
                        min_value=0,
                        max_value=5,
                        value=3,
                        key=f"{dim}_slider_{idx}",
                    )

            st.markdown(f"**Preferred {num_windows} Date Windows**")
            for window_idx in range(1, num_windows + 1):
                cols = st.columns(2)
                with cols[0]:
                    form_state[f"trip_start_{idx}_{window_idx}"] = cols[0].date_input(
                        f"Window {window_idx} - Start Date",
                        value=defaults["start"] + timedelta(weeks=window_idx - 1),
                        key=f"start_input_{idx}_{window_idx}",
                    )
                with cols[1]:
                    form_state[f"trip_end_{idx}_{window_idx}"] = cols[1].date_input(
                        f"Window {window_idx} - End Date",
                        value=defaults["end"] + timedelta(weeks=window_idx - 1),
                        key=f"end_input_{idx}_{window_idx}",
                    )

    submit = st.form_submit_button("🚀 Generate Itinerary", use_container_width=True, type="primary")

if submit:
    try:
        # Validate inputs
        group_members = []
        for idx in range(num_travelers):
            if not form_state[f"name_{idx}"] or not form_state[f"origin_{idx}"]:
                st.error(f"❌ Traveler {idx + 1} needs both name and origin city.")
                st.stop()
            group_members.append(build_member_payload(form_state, idx, num_windows))

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🤖 Initializing AI agent...")
        progress_bar.progress(10)
        
        with st.spinner("🧠 Planning your perfect group getaway... This may take a few minutes."):
            result = run_trip_planner_for_group(group_members)
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.success("✅ Itinerary generated successfully!")
        st.balloons()
        
        structured = result.get("structured_response", {})
        
        # Try to extract JSON from final message if structured is empty
        if not structured or structured == {} or (len(structured) == 1 and "raw_text" in structured):
            final_message = result.get("final_message", "")
            if final_message:
                try:
                    # Try to extract JSON from the message
                    from trip_planner import extract_json_from_markdown
                    extracted = extract_json_from_markdown(final_message)
                    structured = json.loads(extracted)
                except:
                    # If that fails, the reconstruction from tool messages should already be in structured
                    pass

        # Check if we have a valid city (either from parsing or reconstruction)
        if structured and structured.get("chosen_city"):
            chosen_city = structured.get('chosen_city', 'Unknown Destination')
            
            # Show a note if this was reconstructed from tool messages
            if structured.get("explanation") and "reconstructed" in structured.get("explanation", "").lower():
                st.info("ℹ️ Note: This itinerary was automatically reconstructed from agent tool outputs. Some details may be simplified.")
            
            # Header with destination and metrics
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"# 🌍 {chosen_city}")
            with col2:
                if window := structured.get("chosen_trip_window"):
                    st.metric("Trip Dates", f"{window.get('trip_start', '')} to {window.get('trip_end', '')}")
            with col3:
                if fairness := structured.get("fairness_summary"):
                    score = fairness.get('trip_fairness_score', 0)
                    st.metric("Fairness Score", f"{score:.1f}/100")
            
            # Flights Section
            if flight := structured.get("chosen_flight"):
                st.markdown("### ✈️ Flight Information")
                if isinstance(flight, dict):
                    # Check if there's an error message
                    if "error" in flight:
                        st.warning(f"⚠️ {flight.get('error', 'No flights found for this route.')}")
                        st.info("💡 Try adjusting your dates or destination. Some smaller cities may not have direct flights.")
                    else:
                        flight_cols = st.columns(min(len(flight), 3))
                        for i, (name, flight_info) in enumerate(flight.items()):
                            with flight_cols[i % len(flight_cols)]:
                                with st.container():
                                    st.markdown(f"**{name}**")
                                    if isinstance(flight_info, dict):
                                        st.write(f"**Airline:** {flight_info.get('airline', 'N/A')}")
                                        st.write(f"**Price:** ${flight_info.get('price', 'N/A')}")
                                        if 'depart_time' in flight_info:
                                            st.caption(f"Depart: {flight_info.get('depart_time', '')}")
                                        if 'arrive_time' in flight_info:
                                            st.caption(f"Arrive: {flight_info.get('arrive_time', '')}")
                else:
                    st.json(flight)
            else:
                st.markdown("### ✈️ Flight Information")
                st.info("No flight information available. The agent may not have found flights for this route.")
            
            # Hotel Section
            if hotel := structured.get("chosen_hotel"):
                st.markdown("### 🏨 Accommodation")
                if isinstance(hotel, dict):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**{hotel.get('name', 'Hotel')}**")
                    with col2:
                        st.metric("Rating", f"{hotel.get('rating', 0):.1f} ⭐")
                    with col3:
                        st.metric("Price/Night", f"${hotel.get('price_per_night', 0)}")
                else:
                    st.json(hotel)
            
            # Itinerary Section - Beautiful Display
            if itinerary := structured.get("itinerary"):
                st.markdown("---")
                st.markdown("## 📅 Your 2-Day Itinerary")
                
                # Create tabs for each day
                day_tabs = st.tabs([f"Day {i+1}" for i in range(len(itinerary))])
                
                for day_idx, (day_key, day_activities) in enumerate(itinerary.items()):
                    with day_tabs[day_idx]:
                        # Day header
                        day_name = day_key.replace("day", "Day ").title()
                        st.markdown(f"### {day_name}")
                        
                        # Morning
                        if "morning" in day_activities:
                            morning = day_activities["morning"]
                            with st.expander("🌅 Morning", expanded=True):
                                activity_name = morning.get('activity', 'Activity TBD')
                                st.markdown(f"#### {activity_name}")
                                if reasoning := morning.get('reasoning'):
                                    st.info(f"💡 {reasoning}")
                        
                        # Afternoon
                        if "afternoon" in day_activities:
                            afternoon = day_activities["afternoon"]
                            with st.expander("☀️ Afternoon", expanded=True):
                                activity_name = afternoon.get('activity', 'Activity TBD')
                                st.markdown(f"#### {activity_name}")
                                if reasoning := afternoon.get('reasoning'):
                                    st.info(f"💡 {reasoning}")
                        
                        # Evening
                        if "evening" in day_activities:
                            evening = day_activities["evening"]
                            with st.expander("🌙 Evening", expanded=True):
                                activity_name = evening.get('activity', 'Activity TBD')
                                st.markdown(f"#### {activity_name}")
                                if reasoning := evening.get('reasoning'):
                                    st.info(f"💡 {reasoning}")
            
            # Fairness Summary
            if fairness := structured.get("fairness_summary"):
                st.markdown("---")
                st.markdown("## 💰 Budget Fairness Analysis")
                
                score = fairness.get('trip_fairness_score', 0)
                mean_ratio = fairness.get('mean_ratio', 0)
                std_ratio = fairness.get('std_ratio', 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fairness Score", f"{score:.1f}/100")
                with col2:
                    st.metric("Mean Affordability Ratio", f"{mean_ratio:.3f}")
                with col3:
                    st.metric("Cost Variation (STD)", f"{std_ratio:.3f}")
                
                if per_person := fairness.get("per_person"):
                    st.markdown("### Per-Person Breakdown")
                    df_data = []
                    for person in per_person:
                        df_data.append({
                            "Name": person.get("name", ""),
                            "Trip Cost": f"${person.get('trip_cost', 0):.2f}",
                            "Budget": f"${person.get('budget', 0):.2f}",
                            "Affordability Ratio": f"{person.get('affordability_ratio', 0):.3f}",
                            "Status": person.get('affordability_label', '').replace('_', ' ').title()
                        })
                    st.dataframe(df_data, use_container_width=True, hide_index=True)
            
            # Explanation
            if explanation := structured.get("explanation"):
                st.markdown("---")
                with st.expander("📝 Planner Explanation"):
                    st.write(explanation)
        
        else:
            # Fallback: show raw message
            st.warning("⚠️ Could not parse structured itinerary. Showing raw agent reply.")
            with st.expander("Raw Agent Response"):
                st.code(result.get("final_message", ""), language="json")
        
        # Debug section (collapsed)
        with st.expander("🔧 Debug Information"):
            st.json(result.get("agent_result", {}))

    except Exception as err:
        st.error(f"❌ Trip planning failed: {err}")
        st.exception(err)
