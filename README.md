# AI Group Trip Planner

***Made for MadHacks November 2025 at the University of Wisconsin—Madison***

**Live App:** https://ai-squad-planner-fzk9tarzmendo6eg9fynwj.streamlit.app/

An AI-powered trip planning application that helps groups plan fair, budget-aware 2-day trips using Gemini AI and LangChain. The app analyzes group preferences, searches for flights and hotels, and generates personalized itineraries with budget fairness analysis.

## 🎯 Features

- 🎯 **Dynamic Group Size**: Plan trips for 2-10 travelers
- 📅 **Flexible Date Selection**: Each traveler can provide multiple preferred date windows
- ✈️ **Real-time Flight Search**: Integrated SerpAPI for live flight information
- 🏨 **Hotel Recommendations**: Find accommodations based on group budget
- 🎨 **Personalized Activities**: AI-powered activity recommendations using Yelp data
- 💰 **Budget Fairness Analysis**: Ensures trip costs are fair across all group members
- 🤖 **AI-Powered Planning**: Uses Google Gemini AI with LangChain/LangGraph for intelligent itinerary generation

## 📋 Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini AI) - [Get one here](https://makersuite.google.com/app/apikey)
- SerpAPI Key (for flight/hotel searches) - [Get one here](https://serpapi.com/)
- Yelp Academic Dataset (for activity data) - [Download here](https://www.yelp.com/dataset)

## 🚀 Setup Instructions

### Step 1: Clone the Repository

Clone this repository from GitHub and move into the project directory:

```bash
git clone https://github.com/<your-username>/AI-Squad-Planner.git
cd AI-Squad-Planner
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or if you prefer using a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Download Yelp Academic Dataset

**Note:** The Yelp dataset files are NOT included in this repository due to their large size.

1. Go to [Yelp Dataset](https://www.yelp.com/dataset)
2. Sign up/Log in to access the dataset
3. Download the following files:
   - `yelp_academic_dataset_business.json`
   - `yelp_academic_dataset_review.json`
4. Place both files in the project root directory (`AI-Squad-Planner/`)

Your directory should look like:
```
AI-Squad-Planner/
├── yelp_academic_dataset_business.json  ← Download this
├── yelp_academic_dataset_review.json    ← Download this
├── streamlit_app.py
├── trip_planner.py
└── ...
```

### Step 4: Generate Activities CSV

Once you have the Yelp dataset files, generate the `activities.csv` file:

```bash
python create_activities_csv.py
```

This script will:
- Process all cities from the Yelp dataset
- Apply quality filters (min 4.0 stars, min 50 reviews)
- Extract top reviews for each business
- Generate activity tags (nightlife, adventure, shopping, food, urban)
- Create `activities.csv` in the project root

**Processing Time:** This may take 10-30 minutes depending on your system, as it processes millions of records.

**Expected Output:**
```
✓ Selected X businesses across Y cities
✓ Found reviews for Z businesses
✓ Final dataset shape: (X, columns)
✅ Successfully created activities.csv
```

### Step 5: Set Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add your API keys to the `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
SERPAPI_KEY=your_serpapi_key_here
```

**Important Security Notes:**
- ✅ The `.env` file is already in `.gitignore` and will NOT be committed to Git
- ✅ Never commit API keys to GitHub
- ✅ For deployment, use Streamlit Cloud secrets (see Deployment section)

### Step 6: Verify Setup

Check that all required files (such as `activities.csv`, `.env`, `streamlit_app.py`, and `trip_planner.py`) exist in the project root before running the app.

## 🏃 Running the Application

### Local Development

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

1. **Configure Settings** (Sidebar):
   - Set number of travelers (2-10)
   - Set number of date windows per traveler (1-5)

2. **Enter Traveler Information**:
   - Name and origin airport code
   - Total budget
   - Personal preferences/notes
   - Preference weights (0-5) for: nightlife, adventure, shopping, food, urban
   - Preferred date windows

3. **Generate Itinerary**:
   - Click "🚀 Generate Itinerary"
   - Wait for AI to plan your trip (may take 2-5 minutes)
   - View the complete itinerary with flights, hotels, activities, and budget analysis

## 📁 Project Structure

```
SquadPlanner/
├── streamlit_app.py              # Main Streamlit UI application
├── trip_planner.py               # Core trip planning logic with LangChain agent
├── create_activities_csv.py      # Script to generate activities.csv from Yelp data
├── activities.csv                # Generated activity data (created in Step 4)
├── requirements.txt              # Python package dependencies
├── .gitignore                    # Git ignore rules (protects .env and data files)
├── README.md                     # This file
│
├── yelp_academic_dataset_business.json  # Yelp dataset (download separately)
└── yelp_academic_dataset_review.json    # Yelp dataset (download separately)
```

## 🔧 Configuration

### Activities CSV Generation

You can customize the `create_activities_csv.py` script:

```python
# Quality filters
MIN_STARS = 4.0                    # Minimum star rating
MIN_REVIEW_COUNT = 50             # Minimum number of reviews
MAX_BUSINESSES_PER_CITY = 150     # Max businesses per city

# Optional: Filter specific cities
CITIES = None  # Process all cities
# Or specify: CITIES = {"New Orleans", "Philadelphia", "Tucson"}
```

### Trip Planner Settings

In `trip_planner.py`, you can adjust:
- AI model temperature
- Number of activities to consider
- Budget fairness calculation parameters

## 🌐 Deployment to Streamlit Cloud

### Step 1: Prepare Your Repository

Make sure your code is on GitHub:

```bash
# Initialize git (if not already done)
git init

# Add files (will exclude .env and data files automatically)
git add .

# Commit
git commit -m "Initial commit: AI Group Trip Planner"

# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Important:** Make sure `activities.csv` is committed to the repository (it should be, as it's not in `.gitignore`).

### Step 2: Deploy on Streamlit Cloud

To deploy your own instance on Streamlit Community Cloud:

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select the repository and branch containing this project
5. Set **Main file path** to `streamlit_app.py`
6. In **Advanced settings → Secrets**, add your keys in TOML format:

```toml
GOOGLE_API_KEY = "<your-google-api-key>"
SERPAPI_KEY = "<your-serpapi-key>"
```

7. Click **"Deploy"**

### Step 3: Verify Deployment

- Your app will be available at a URL like `https://your-app-name.streamlit.app`
- Check the logs if there are any errors
- Make sure `activities.csv` is accessible (it should be in the repo)

## 🔒 Security Best Practices

- ✅ **Never commit API keys**: The `.env` file is in `.gitignore`
- ✅ **Use environment variables**: All API keys are loaded from environment variables
- ✅ **Streamlit Cloud secrets**: Use the secrets feature for deployment
- ✅ **No hardcoded keys**: All sensitive data is externalized

## 🐛 Troubleshooting

### "GOOGLE_API_KEY not set"
- **Solution**: Make sure your `.env` file exists and contains `GOOGLE_API_KEY=your_key`
- Check that `python-dotenv` is installed: `pip install python-dotenv`

### "activities.csv not found"
- **Solution**: Run `python create_activities_csv.py` first
- Make sure the Yelp dataset JSON files are in the project root
- Check that the script completed successfully

### "SERPAPI_KEY not found"
- **Solution**: Add `SERPAPI_KEY=your_key` to your `.env` file
- Verify your SerpAPI key is valid and has credits

### Flight/Hotel search returns errors
- **Solution**: 
  - Check your SerpAPI key is valid
  - Verify you have credits in your SerpAPI account
  - Some smaller cities may not have flight data available

### "No flights found for this route"
- **Solution**: This is expected for some destinations. The app will try alternative cities or show a message.

### Activities CSV generation is slow
- **Solution**: This is normal - processing millions of Yelp records takes time (10-30 minutes)
- The script shows progress updates
- You can filter to specific cities in `create_activities_csv.py` to speed it up

### Import errors
- **Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

## 📊 Data Sources

- **Activities Data**: Yelp Academic Dataset
- **Flight Data**: SerpAPI (Google Flights)
- **Hotel Data**: SerpAPI (Google Hotels)
- **AI Model**: Google Gemini 2.5 Flash

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Yelp for providing the academic dataset
- Google for Gemini AI
- LangChain/LangGraph for the agent framework
- Streamlit for the web framework

## 📧 Support

If you encounter any issues:
1. Check the Troubleshooting section above
2. Review the error messages in the Streamlit app logs
3. Verify all setup steps were completed correctly
4. Ensure all API keys are valid and have sufficient credits

---

**Happy Trip Planning! ✈️🌍**
