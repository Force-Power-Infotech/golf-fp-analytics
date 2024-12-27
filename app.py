import streamlit as st
import pandas as pd
from document_processor import process_document, validate_api_key, analyze_player_performance
from ppt_generator import create_presentation
import tempfile
import os
from config import OPENAI_API_KEY

def main():
    try:
        # Basic page config
        st.set_page_config(page_title="Golf Player Analyzer", layout="wide")
        
        # Initialize session state
        if 'data_df' not in st.session_state:
            st.session_state.data_df = None
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
        if 'players_list' not in st.session_state:
            st.session_state.players_list = None

        # Main interface
        st.title("⛳ Golf Player Performance Analyzer")
        st.markdown("### Upload player statistics file for analysis")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Excel/CSV file with player statistics",
            type=['xlsx', 'csv'],
            key='file_uploader'
        )
        
        if uploaded_file is not None:
            try:
                # Load and store data in session state
                if st.session_state.data_df is None:
                    with st.spinner('Loading data...'):
                        if uploaded_file.name.endswith('.xlsx'):
                            df = pd.read_excel(uploaded_file)
                        else:
                            df = pd.read_csv(uploaded_file)
                        st.session_state.data_df = df
                        
                        # Get players list
                        try:
                            player_col = next(col for col in df.columns if 'player' in col.lower())
                            st.session_state.players_list = sorted(df[player_col].unique().tolist())
                        except StopIteration:
                            st.error("Could not find player column in the data")
                            return
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(st.session_state.data_df.head())
                
                # Player selection
                if st.session_state.players_list:
                    selected_player = st.selectbox(
                        "Select a player to analyze:",
                        options=st.session_state.players_list,
                        key='player_selector'
                    )
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if st.button("Analyze Player", key='analyze_btn'):
                            with st.spinner('Analyzing player performance...'):
                                # Get API client
                                is_valid, client = validate_api_key(OPENAI_API_KEY)
                                if not is_valid:
                                    st.error("Invalid API key")
                                    return
                                
                                # Perform analysis
                                analysis_result = analyze_player_performance(
                                    client, 
                                    st.session_state.data_df, 
                                    selected_player
                                )
                                st.session_state.analysis_result = analysis_result
                    
                    with col2:
                        if st.session_state.analysis_result:
                            st.info("Analysis complete! See results below.")
                
                # Display results
                if st.session_state.analysis_result:
                    st.subheader("🔍 Analysis Results")
                    st.write(st.session_state.analysis_result)
                    
                    # Generate and offer PPT download
                    if not st.session_state.analysis_result.startswith('Error'):
                        try:
                            with st.spinner('Generating presentation...'):
                                ppt_path = create_presentation(str(st.session_state.analysis_result))
                                if os.path.exists(ppt_path):
                                    with open(ppt_path, "rb") as ppt:
                                        ppt_data = ppt.read()
                                        st.download_button(
                                            label="📥 Download Analysis PPT",
                                            data=ppt_data,
                                            file_name=f"{selected_player}_analysis.pptx",
                                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                                        )
                                    # Clean up temporary file
                                    os.unlink(ppt_path)
                        except Exception as e:
                            st.error(f"Error generating presentation: {str(e)}")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        

        # Add footer
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload your Excel/CSV file containing player statistics
        2. Select a player from the dropdown menu
        3. Click "Analyze Player" to generate insights
        4. Review the analysis and download the presentation
        """)

        # Add version footer
        st.markdown("---")
        st.markdown("**Version 1.0.4**")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page and try again")

if __name__ == "__main__":
    main()
