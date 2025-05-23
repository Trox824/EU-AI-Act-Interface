# AI App Review Analyzer with EU AI Act Classification

This application analyzes Google Play Store reviews for AI apps, compares user feedback with developer descriptions, and performs risk classification according to the EU AI Act criteria.

## Features

- Search for AI apps on Google Play Store
- Retrieve app details and user reviews
- Filter reviews by length
- Analyze reviews with OpenAI to extract key insights
- Compare user feedback with developer claims
- Classify apps according to EU AI Act risk categories
- Display detailed logs of each step

## EU AI Act Classification

The application includes a feature to automatically classify apps according to the EU AI Act risk categories:

- **Unacceptable risk:** AI systems that pose a clear threat to people's safety, livelihoods, or rights. These are prohibited.
- **High risk:** AI systems with significant potential for harm to health, safety, fundamental rights, environment, democracy, or rule of law.
- **Limited risk:** AI systems with specific transparency obligations, like disclosing that content is AI-generated.
- **Minimal Risk:** AI systems with minimal or no risk that are not specifically regulated.

The classification is performed using OpenAI's GPT-4.1-nano model with structured output parsing for more reliable results, working through a series of assessment questions for each risk category.

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
  │   ├── input.py         # Input form components
  │   └── analysis.py      # Analysis display
  └── utils/               # Utility functions
      └── data_utils.py    # Data manipulation utilities
/asset
  └── EU_AI_Act_Assessment_Questions.csv  # EU AI Act assessment criteria
```

## Deploying to Streamlit Cloud

1. Push to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Add your OpenAI API key to the Streamlit Cloud secrets through the app settings

## License

[MIT](LICENSE)
