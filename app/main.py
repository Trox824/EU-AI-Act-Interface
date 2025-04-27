"""
Main entry point for the AI App Review Analyzer.
"""
import streamlit as st

from config.settings import APP_TITLE, APP_DESCRIPTION, LAYOUT
from services.logger import logger
from services.playstore import PlayStoreService
from services.analysis import AnalysisService
from ui.search import search_input, debounced_search, display_app_list
from ui.status import show_analysis_log
from ui.analysis import display_analysis_results, display_footer, display_app_details_table
from ui.custom import load_custom_css, styled_header

def run_analysis(app_id, app_name, analysis_service, playstore_service, **kwargs):
    # Get app details and reviews
    app_details, reviews_df = playstore_service.get_app_details_and_reviews(
        app_id, app_name, status_logger=kwargs.get('status_logger')
    )
    
    # Analyze the app
    return analysis_service.analyze_app(
        app_details, reviews_df, status_logger=kwargs.get('status_logger')
    )

def main():
    """Main entry point for the application."""
    # Set up page configuration
    st.set_page_config(page_title=APP_TITLE, layout=LAYOUT)
    
    # Load custom CSS
    load_custom_css()
    
    # Display page title and description
    styled_header(APP_TITLE, level=1)
    st.caption(APP_DESCRIPTION)
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = [] # Now a list of dicts
    if 'selected_app_name' not in st.session_state:
        st.session_state.selected_app_name = None
    if 'selected_app_id' not in st.session_state:
        st.session_state.selected_app_id = None
    if 'selected_app_details' not in st.session_state: # Add state for fetched details
        st.session_state.selected_app_details = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'analysis_triggered' not in st.session_state: # Track if analysis button was clicked
        st.session_state.analysis_triggered = False
    if 'playstore_service' not in st.session_state:
        st.session_state.playstore_service = PlayStoreService()
    
    main_container = st.container() 
    
    with main_container:
        try:
            analysis_service = AnalysisService()
            playstore_service = st.session_state.playstore_service
            col1, col2 = st.columns([1, 1])

            # --- Left Column (Search & Select) ---
            with col1:
                st.subheader("Search & Select App")
                query = search_input(st.session_state.last_query)
                
                # Auto-search with debounce when query changes
                if query != st.session_state.last_query:
                    st.session_state.last_query = query
                    st.session_state.selected_app_name = None
                    st.session_state.selected_app_id = None
                    st.session_state.analysis_results = None
                    st.session_state.search_results = [] 
                    st.session_state.analysis_triggered = False # Reset trigger
                    
                    if query:
                        st.session_state.search_results = debounced_search(query)
                        st.rerun()
                    else: # Clear results if query is empty
                        st.rerun()

                # Display app list if search results exist
                if st.session_state.search_results:
                    app_name, app_id = display_app_list(st.session_state.search_results)
                    
                    # Handle app selection via button click
                    if app_name is not None and app_id is not None:
                        # Check if the selection actually changed
                        if (app_name != st.session_state.selected_app_name or 
                            app_id != st.session_state.selected_app_id):
                            
                            st.session_state.selected_app_name = app_name
                            st.session_state.selected_app_id = app_id
                            st.session_state.analysis_results = None # Clear old analysis results
                            st.session_state.analysis_triggered = False # Reset trigger
                            st.session_state.selected_app_details = None # Clear old details first
                            
                            # Fetch details immediately
                            with st.spinner("Fetching app details..."): 
                                app_details = playstore_service.get_app_details(app_id, app_name)
                            st.session_state.selected_app_details = app_details
                            
                            st.rerun() # Rerun to display details table

            # --- Right Column (Details, Log & Analysis) ---
            with col2:
                st.subheader("Analysis Details")
                
                # Display details table if available
                if st.session_state.selected_app_details:
                    display_app_details_table(st.session_state.selected_app_details)
                    st.divider() # Add a separator
                
                # Show analysis button only if details are loaded (meaning an app is selected)
                if st.session_state.selected_app_details:
                    # Display analysis button
                    if st.button("Analyze Selected App Reviews"):
                        st.session_state.analysis_triggered = True
                        st.session_state.analysis_results = None # Clear previous display
                        st.rerun() # Rerun to trigger analysis log display
                elif not st.session_state.selected_app_id and not st.session_state.search_results:
                    # Show initial message only if no search happened yet
                    st.info("Search for an app on the left to get started.")
                elif not st.session_state.selected_app_id:
                    # Show message if search happened but no selection yet
                    st.info("Select an app from the list on the left to see its details.")

                # Run analysis and show log if triggered
                if st.session_state.analysis_triggered and st.session_state.selected_app_id:
                    analysis_results = show_analysis_log(
                        st.session_state.selected_app_name, 
                        run_analysis, 
                        st.session_state.selected_app_id,    
                        st.session_state.selected_app_name,  
                        analysis_service=analysis_service,
                        playstore_service=playstore_service
                    )
                    st.session_state.analysis_results = analysis_results
                    st.session_state.analysis_triggered = False 

                # Display analysis results if available
                if st.session_state.analysis_results:
                    display_analysis_results(
                        st.session_state.analysis_results,
                        st.session_state.selected_app_name
                    )

        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {e}")
        finally:
            # Make sure to clean up Selenium resources when the app is done
            if 'playstore_service' in st.session_state and st.session_state.playstore_service:
                st.session_state.playstore_service.close_selenium_scraper()
    
    # Call footer at the very end
    display_footer()

if __name__ == "__main__":
    main() 