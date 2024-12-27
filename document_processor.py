import pandas as pd
import docx
from openai import OpenAI
import os
from openai import OpenAIError
import tiktoken
from pptx import Presentation
from config import OPENAI_API_KEY
import json  # Add this import at the top with other imports
import numpy as np  # Add this import at the top with other imports
import datetime  # Add this import at the top with other imports


def validate_api_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True, client
    except OpenAIError as e:
        print(f"API Key validation failed: {str(e)}")
        return False, None


def read_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text


def read_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_string()


def get_token_count(text, model="gpt-3.5-turbo"):
    """Count tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def chunk_content(text, max_tokens=14000):
    """Split content into chunks that fit within token limits."""
    chunks = []
    current_chunk = ""
    current_tokens = 0

    # Split text into paragraphs
    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        paragraph_tokens = get_token_count(paragraph)

        # If single paragraph exceeds limit, split it into smaller pieces
        if paragraph_tokens > max_tokens:
            words = paragraph.split()
            temp_chunk = ""
            for word in words:
                if get_token_count(temp_chunk + " " + word) < max_tokens:
                    temp_chunk += " " + word
                else:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = word
            if temp_chunk:
                chunks.append(temp_chunk.strip())
            continue

        # Check if adding paragraph exceeds limit
        if current_tokens + paragraph_tokens < max_tokens:
            current_chunk += "\n" + paragraph
            current_tokens += paragraph_tokens
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
            current_tokens = paragraph_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def analyze_chunk(client, chunk, template_info=None):
    """Analyze content with reference to template structure."""
    try:
        system_prompt = """You are a professional business analyst. Provide a comprehensive analysis with the following structure, ensuring all numerical data is clearly formatted:

1. Executive Summary
2. Key Metrics (each on new line, strictly in format "MetricName: Number" or "Category: XX%"):
   - Revenue: 1234567
   - Growth Rate: 25%
   - Market Share: 45%
   - Customer Count: 5000
3. Trend Analysis (each on new line, in format "Trend: Number"):
   - Q1 Growth Trend: 15
   - Q2 Growth Trend: 25
   - Q3 Growth Trend: 35
4. Segment Analysis (each on new line, in format "Segment: Number"):
   - Enterprise Segment: 45
   - SMB Segment: 30
   - Consumer Segment: 25
5. Performance Metrics (each on new line, in format "Metric: Number"):
   - Sales Performance: 85
   - Customer Satisfaction: 92
   - Market Penetration: 78
6. Recommendations (each with impact percentage):
   - Recommendation 1 (Impact: 30%)
   - Recommendation 2 (Impact: 25%)

Ensure EVERY numerical value is presented in the exact format specified above for proper chart generation."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"Error analyzing chunk: {str(e)}"


def get_players_list(file_path):
    """Extract list of players from the Excel/CSV file."""
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        # Assuming player names are in a column named 'Player' or similar
        player_column = next(
            col for col in df.columns if 'player' in col.lower())
        players = df[player_column].unique().tolist()
        return players
    except Exception as e:
        print(f"Error getting players list: {str(e)}")
        return []


def get_player_column(df):
    """Identify the player column name in the dataframe."""
    possible_names = ['Player', 'player', 'Name',
                      'name', 'PLAYER', 'PlayerName', 'Player Name']

    for name in possible_names:
        if name in df.columns:
            return name

    # If no exact match, try partial matches
    for col in df.columns:
        if any(name.lower() in col.lower() for name in ['player', 'name']):
            return col

    raise ValueError("Could not find player column in the data")


def fetch_data(df, selected_player):
    """Fetch data for the selected player and handle null values."""
    player_data = df[df['Player_Name'] == selected_player].fillna(0)
    # Ensure all relevant columns are present and handle missing columns
    required_columns = ['Tot_par', 'Handicap', 'Game_Result'] + [f'H_{
        i}_GS' for i in range(1, 19)] + [f'H_{i}_NS' for i in range(1, 19)] + ['Tee Time']
    for col in required_columns:
        if col not in player_data.columns:
            player_data[col] = 0
    return player_data


def calculate_analytics(player_data):
    """Calculate all kinds of analytics for the player."""
    hole_index_data = [
        {"hole_no": 1, "par": 4, "stroke_index": 11},
        {"hole_no": 2, "par": 3, "stroke_index": 17},
        {"hole_no": 3, "par": 4, "stroke_index": 3},
        {"hole_no": 4, "par": 5, "stroke_index": 13},
        {"hole_no": 5, "par": 4, "stroke_index": 9},
        {"hole_no": 6, "par": 4, "stroke_index": 5},
        {"hole_no": 7, "par": 4, "stroke_index": 1},
        {"hole_no": 8, "par": 4, "stroke_index": 15},
        {"hole_no": 9, "par": 4, "stroke_index": 7},
        {"hole_no": 10, "par": 4, "stroke_index": 2},
        {"hole_no": 11, "par": 4, "stroke_index": 8},
        {"hole_no": 12, "par": 4, "stroke_index": 14},
        {"hole_no": 13, "par": 3, "stroke_index": 16},
        {"hole_no": 14, "par": 4, "stroke_index": 4},
        {"hole_no": 15, "par": 5, "stroke_index": 12},
        {"hole_no": 16, "par": 4, "stroke_index": 18},
        {"hole_no": 17, "par": 4, "stroke_index": 10},
        {"hole_no": 18, "par": 4, "stroke_index": 6}
    ]
    analytics = {}

    # Convert all relevant columns to numeric, handling errors by setting invalid parsing to NaN
    numeric_columns = ['Tot_par', 'Handicap', 'Game_Result'] + \
        [f'H_{i}_GS' for i in range(1, 19)]
    player_data[numeric_columns] = player_data[numeric_columns].apply(
        pd.to_numeric, errors='coerce').fillna(0)

    # Convert 'Tee Time' to datetime.time for proper comparison
    def convert_to_time(time_str):
        try:
            return pd.to_datetime(time_str, format='%H:%M').time()
        except ValueError:
            return pd.to_datetime(time_str, format='%H:%M:%S').time()

    player_data['Tee Time'] = player_data['Tee Time'].apply(convert_to_time)

    # Overall Performance Metrics
    analytics['Total_Par'] = player_data['Tot_par'].sum()
    analytics['Handicap_Index'] = player_data['Handicap'].mean()
    analytics['Strokes_Gained_vs_Handicap_Group'] = player_data['Game_Result'].mean()
    analytics['Scoring_Average'] = player_data[[
        f'H_{i}_GS' for i in range(1, 19)]].mean().mean()

    # Time of Day Performance Analysis
    am_rounds = player_data[player_data['Tee Time'] <
                            pd.to_datetime('10:00', format='%H:%M').time()]
    pm_rounds = player_data[player_data['Tee Time'] >=
                            pd.to_datetime('10:00', format='%H:%M').time()]
    analytics['AM_Scoring_Average'] = am_rounds[[
        f'H_{i}_GS' for i in range(1, 19)]].mean().mean()
    analytics['PM_Scoring_Average'] = pm_rounds[[
        f'H_{i}_GS' for i in range(1, 19)]].mean().mean()
    analytics['Optimal_Playing_Window'] = 'Morning' if analytics['AM_Scoring_Average'] < analytics['PM_Scoring_Average'] else 'Afternoon'
    analytics['Performance_Delta'] = abs(
        analytics['AM_Scoring_Average'] - analytics['PM_Scoring_Average'])

    # Technical Handicap Analysis
    analytics['Handicap_Peer_Group'] = f"{player_data['Handicap'].min()} - {player_data['Handicap'].max()}"
    analytics['Statistical_Peer_Group_Size'] = len(player_data)
    analytics['Strokes_Gained_Lost_vs_Peer_Group'] = player_data['Game_Result'].mean()

    # Hole-by-Hole Statistical Breakdown
    hole_stats = {}
    for hole in range(1, 19):
        par = next(item['par'] for item in hole_index_data if item['hole_no'] == hole)
        hole_stats[f'Hole_{hole}'] = {
            'Par': player_data[f'H_{hole}_GS'].mean(),
            'Total_Pars': (player_data[f'H_{hole}_GS'] == par).sum(),
            'Double_Bogeys_or_Worse': (player_data[f'H_{hole}_GS'] >= par + 2).sum(),
            'Birdies': (player_data[f'H_{hole}_GS'] == par - 1).sum(),
            'Eagles': (player_data[f'H_{hole}_GS'] <= par - 2).sum(),
            'Morning_Stats': {
                'Average_Score': am_rounds[f'H_{hole}_GS'].mean(),
                'Total_Pars': (am_rounds[f'H_{hole}_GS'] == par).sum(),
                'Double_Bogeys_or_Worse': (am_rounds[f'H_{hole}_GS'] >= par + 2).sum(),
                'Birdies': (am_rounds[f'H_{hole}_GS'] == par - 1).sum(),
                'Eagles': (am_rounds[f'H_{hole}_GS'] <= par - 2).sum()
            },
            'Afternoon_Stats': {
                'Average_Score': pm_rounds[f'H_{hole}_GS'].mean(),
                'Total_Pars': (pm_rounds[f'H_{hole}_GS'] == par).sum(),
                'Double_Bogeys_or_Worse': (pm_rounds[f'H_{hole}_GS'] >= par + 2).sum(),
                'Birdies': (pm_rounds[f'H_{hole}_GS'] == par - 1).sum(),
                'Eagles': (pm_rounds[f'H_{hole}_GS'] <= par - 2).sum()
            }
        }
    analytics['Hole_by_Hole_Stats'] = hole_stats

    # Round-by-Round Performance
    round_stats = []
    for _, row in player_data.iterrows():
        gross_scores = row[[f'H_{i}_GS' for i in range(1, 19)]]
        par_scores = [next(item['par'] for item in hole_index_data if item['hole_no'] == i) for i in range(1, 19)]
        round_stats.append({
            'Date': row['Date'],
            'Tee_Time': row['Tee Time'],
            'Gross_Score': gross_scores.sum(),
            'Total_Pars': (gross_scores == par_scores).sum(),
            'Total_Bogeys': (gross_scores == [par + 1 for par in par_scores]).sum(),
            'Total_Double_Bogeys_or_Worse': (gross_scores >= [par + 2 for par in par_scores]).sum(),
            'Birdies': (gross_scores == [par - 1 for par in par_scores]).sum(),
            'Eagles': (gross_scores <= [par - 2 for par in par_scores]).sum()
        })
    analytics['Round_by_Round_Stats'] = round_stats

    # Add all available data for the selected player for hole-by-hole information
    analytics['Player_Hole_Data'] = player_data.to_dict(orient='records')

    return analytics


def analyze_player_performance(client, df, selected_player):
    try:
        player_data = fetch_data(df, selected_player)
        analytics = calculate_analytics(player_data)

        # Convert all numeric values and datetime objects to standard Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, (datetime.time, datetime.date)):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        analytics = convert_to_serializable(analytics)

        hole_index_data = [
            {"hole_no": 1, "par": 4, "stroke_index": 11},
            {"hole_no": 2, "par": 3, "stroke_index": 17},
            {"hole_no": 3, "par": 4, "stroke_index": 3},
            {"hole_no": 4, "par": 5, "stroke_index": 13},
            {"hole_no": 5, "par": 4, "stroke_index": 9},
            {"hole_no": 6, "par": 4, "stroke_index": 5},
            {"hole_no": 7, "par": 4, "stroke_index": 1},
            {"hole_no": 8, "par": 4, "stroke_index": 15},
            {"hole_no": 9, "par": 4, "stroke_index": 7},
            {"hole_no": 10, "par": 4, "stroke_index": 2},
            {"hole_no": 11, "par": 4, "stroke_index": 8},
            {"hole_no": 12, "par": 4, "stroke_index": 14},
            {"hole_no": 13, "par": 3, "stroke_index": 16},
            {"hole_no": 14, "par": 4, "stroke_index": 4},
            {"hole_no": 15, "par": 5, "stroke_index": 12},
            {"hole_no": 16, "par": 4, "stroke_index": 18},
            {"hole_no": 17, "par": 4, "stroke_index": 10},
            {"hole_no": 18, "par": 4, "stroke_index": 6}
        ]

        # Define the single prompt for the entire analysis
        prompt = f"""
        Provide a comprehensive technical analysis for the selected player with the following sections:

        Player: {selected_player}

        Provided Hole Index Data:
        {json.dumps(hole_index_data, indent=4)}

        1. Overall Performance Metrics:
           - Total Par: {analytics['Total_Par']}
           - Handicap Index: {analytics['Handicap_Index']}
           - Strokes Gained vs Handicap Group: {analytics['Strokes_Gained_vs_Handicap_Group']}
           - Scoring Average: {analytics['Scoring_Average']}

        2. Time of Day Performance Analysis:
           - AM Scoring Average: {analytics['AM_Scoring_Average']}
           - PM Scoring Average: {analytics['PM_Scoring_Average']}
           - Optimal Playing Window: {analytics['Optimal_Playing_Window']}
           - Performance Delta: {analytics['Performance_Delta']} strokes

        3. Technical Handicap Analysis:
           - Handicap Peer Group: {analytics['Handicap_Peer_Group']}
           - Statistical Peer Group Size: {analytics['Statistical_Peer_Group_Size']} players
           - Strokes Gained/Lost vs Peer Group: {analytics['Strokes_Gained_Lost_vs_Peer_Group']}
           - Comparative shot pattern analysis
           - Scoring distribution on par 3s/4s/5s
           - Course management efficiency rating

        4. Hole-by-Hole Statistical Breakdown:
           {json.dumps(analytics['Hole_by_Hole_Stats'], indent=4)}
           - FORMAT AS: "Hole 1 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 2 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 3 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 4 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 5 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 6 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 7 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 8 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 9 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 10 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 11 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 12 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 13 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 14 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 15 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 16 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 17 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
           - FORMAT AS: "Hole 18 ([Par], Stroke Index: [stroke_index]): [player score] (Peer Avg: [avg], Field Avg: [avg])"
            - FORMAT AS: "Total Pars: [number]"
            - FORMAT AS: "Double Bogeys or Worse: [number]"
            - Morning Stats:
            - Afternoon Stats:
            - Risk/reward decision points
            - Shot distribution patterns
            - Critical scoring opportunities
            - Recovery shot efficiency

        5. Round-by-Round Performance:
           {json.dumps(analytics['Round_by_Round_Stats'], indent=4)}
           - FORMAT AS: "[Date]:
                - Gross Score: [score] [handicap added if applicable]
                - Total Pars: [pars]
                - Total Bogeys: [bogeys]
                - Total Double Bogeys or Worse: [dbw]"

        6. Key Performance Insights:
           - Strategic Finding: [detailed description]
           - Time-based performance variations including:
             * Morning vs Afternoon par conversion rates
             * Time-specific double bogey patterns
             * Scoring distribution by time of day
           - Course management decisions
           - Scoring pattern anomalies
           - Statistical strengths/weaknesses
           - Weather impact correlations

        7. Professional Development Recommendations:
           - Technical Recommendation: [specific action] (Projected Impact: XX%)
           - Optimal tee time strategy
           - Shot selection modifications
           - Practice priority areas
           - Course management adjustments
           - Environmental adaptation strategies

        Technical Analysis Requirements:
        - Emphasize strokes gained/lost metrics
        - Include detailed morning vs afternoon statistical comparisons
        - Analyze scoring patterns relative to playing conditions
        - Evaluate decision-making efficiency
        - Quantify performance under varying conditions
        - Provide specific practice protocols

        Player Hole Data:
        {json.dumps(analytics['Player_Hole_Data'], indent=4)}
        """

        # Call the API with the single prompt
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional golf analyst specializing in statistical analysis and performance improvement recommendations.Provide detailed results for every item in the data but for numbers, provide up to two decimal places if applicable. Do not summarize or condense the output. Include all repetitive patterns explicitly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.75,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing player performance: {str(e)}"

    except Exception as e:
        return f"Error analyzing player performance: {str(e)}\nDataframe columns: {', '.join(df.columns.tolist())}"


def process_document(file_path, selected_player=None):
    try:
        is_valid, client = validate_api_key(OPENAI_API_KEY)
        if not is_valid:
            return "Error: Invalid OpenAI API key. Please check the configuration."

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in ['.xlsx', '.csv']:
            try:
                if file_extension == '.xlsx':
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)

                if selected_player:
                    # Perform player-specific analysis
                    return analyze_player_performance(client, df, selected_player)
                else:
                    # Return list of players
                    return get_players_list(file_path)

            except Exception as e:
                return f"Error processing file: {str(e)}"
        else:
            return f"Unsupported file format: {file_extension}"

    except Exception as e:
        return f"Error processing document: {str(e)}"
