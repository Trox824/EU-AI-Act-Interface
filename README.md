# AI App Review Analyzer

This application analyzes Google Play Store reviews for AI apps and compares user feedback with developer descriptions.

## Features

- Search for AI apps on Google Play Store
- Retrieve app details and user reviews
- Filter reviews by length
- Analyze reviews with OpenAI to extract key insights
- Compare user feedback with developer claims
- Display detailed logs of each step

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd EUAIActInteface
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a file `.streamlit/secrets.toml` in the project directory
   - Add the following line to the file:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```
   - Replace `"your-api-key-here"` with your actual OpenAI API key

## Running the App

Run the Streamlit app with:

```bash
streamlit run app/main.py
```

## Project Structure

The project follows a modular architecture:

```
/app
  ├── main.py              # Main entry point
  ├── config/              # Configuration settings
  │   └── settings.py      # App configuration constants
  ├── models/              # Data models
  │   └── app_data.py      # Data structures
  ├── services/            # Business logic
  │   ├── playstore.py     # Play Store API interaction
  │   ├── analysis.py      # OpenAI analysis logic
  │   └── logger.py        # Logging setup
  ├── ui/                  # UI components
  │   ├── search.py        # Search interface
  │   ├── analysis.py      # Analysis display
  │   └── status.py        # Status display
  └── utils/               # Utility functions
      └── data_utils.py    # Data manipulation utilities
```

## Deploying to Streamlit Cloud

1. Push to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Add your OpenAI API key to the Streamlit Cloud secrets through the app settings

## License

[MIT](LICENSE)
