import streamlit as st
import pickle
import pandas as pd
import numpy as np
import gdown
import os
import plotly.express as px
import base64

# --- 1. Page Configuration & Custom CSS ---
st.set_page_config(page_title="T20 Predictor Pro", page_icon="🏏", layout="wide")

# Helper function to load local images as base64 for CSS/HTML injection
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

# Load your stadium background
bg_image = get_base64_of_bin_file("stadium.jpg") 

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;800&display=swap');
    
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(15, 23, 42, 0.85); 
        padding: 3rem !important;
        border-radius: 20px;
        margin-top: 3rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px); 
        border: 1px solid rgba(255, 255, 255, 0.18);
    }}
    
    /* Custom Centered Title with Premium Font */
    .main-title {{
        font-family: 'Poppins', sans-serif;
        text-align: center;
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0px;
        padding-bottom: 0px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .sub-title {{
        text-align: center;
        color: #adb5bd !important;
        font-size: 18px;
        margin-top: 5px;
        margin-bottom: 30px;
        font-family: 'Helvetica Neue', sans-serif;
    }}
    
    h1, h2, h3, p, span, div {{
        color: #f8f9fa !important;
    }}
    .subheader {{
        color: #ffc107 !important; 
        font-weight: 600;
        margin-top: 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 10px;
    }}
    .metric-container {{
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }}
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div {{
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
    }}
    .footer {{
        text-align: center;
        margin-top: 50px;
        color: #adb5bd !important;
        font-size: 14px;
        letter-spacing: 1px;
    }}
    </style>
""", unsafe_allow_html=True)

# --- 2. Load Model ---
@st.cache_resource
def load_model():
    file_id = "15N4KPQc7Job-26w3fKxVfqHCK5y_Pq0w"
    if not os.path.exists("pipe.pkl"):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "pipe.pkl", quiet=False)
    return pickle.load(open("pipe.pkl","rb"))

pipe = load_model()

# --- 3. Data Setup & Flags Dictionary ---
teams = ['Australia','India','Bangladesh','New Zealand','South Africa',
         'England','West Indies','Afghanistan','Pakistan','Sri Lanka']

cities = ['Colombo','Mirpur','Johannesburg','Dubai','Auckland','Cape Town',
          'London','Pallekele','Barbados','Sydney','Melbourne','Durban',
          'St Lucia','Wellington','Lauderhill','Hamilton','Centurion',
          'Manchester','Abu Dhabi','Mumbai','Nottingham','Southampton',
          'Mount Maunganui','Chittagong','Kolkata','Lahore','Delhi',
          'Nagpur','Chandigarh','Adelaide','Bangalore','St Kitts',
          'Cardiff','Christchurch','Trinidad']

flags = {
    'Australia': 'https://flagcdn.com/w160/au.png',
    'India': 'https://flagcdn.com/w160/in.png',
    'Bangladesh': 'https://flagcdn.com/w160/bd.png',
    'New Zealand': 'https://flagcdn.com/w160/nz.png',
    'South Africa': 'https://flagcdn.com/w160/za.png',
    'England': 'https://flagcdn.com/w160/gb-eng.png',
    'West Indies': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_the_West_Indies_cricket_team.svg/200px-Flag_of_the_West_Indies_cricket_team.svg.png',
    'Afghanistan': 'https://flagcdn.com/w160/af.png',
    'Pakistan': 'https://flagcdn.com/w160/pk.png',
    'Sri Lanka': 'https://flagcdn.com/w160/lk.png'
}

# --- 4. Hero Section (Centered Titles with Image) ---
# Load your newly generated transparent image
t20_logo = get_base64_of_bin_file("t20_logo_transparent.png") 

# Inject it directly into the h1 tag (No mix-blend-mode hack!)
st.markdown(f"""
    <h1 class='main-title'>
        <img src='data:image/png;base64,{t20_logo}' width='140' style='margin-right: 20px;'/>
        SCORE PREDICTOR
    </h1>
""", unsafe_allow_html=True)

st.markdown("<p class='sub-title'>Advanced Machine Learning Engine to project the final innings score.</p>", unsafe_allow_html=True)


# --- 5. Main Screen Match Setup ---
st.markdown("<h3 class='subheader'>Match Setup</h3>", unsafe_allow_html=True)

setup_col1, setup_col2, setup_col3 = st.columns(3)
with setup_col1:
    batting_team = st.selectbox('Batting Team', sorted(teams), index=sorted(teams).index('India'))
with setup_col2:
    bowling_team = st.selectbox('Bowling Team', sorted(teams), index=sorted(teams).index('Australia'))
with setup_col3:
    city = st.selectbox('Match City', sorted(cities), index=sorted(cities).index('Melbourne'))

# --- 6. Interface & Core Logic ---
if batting_team == bowling_team:
    st.error("Batting and Bowling teams must be different!")
else:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Dynamic Flags UI ---
    flag_col1, vs_col, flag_col2 = st.columns([1, 1, 1])
    with flag_col1:
        st.markdown(f"<div style='text-align: right;'><img src='{flags[batting_team]}' width='140' style='border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);'/></div>", unsafe_allow_html=True)
    with vs_col:
        st.markdown("<h1 style='text-align: center; color: #ffc107; font-size: 3rem; margin-top: 10px; font-family: \"Poppins\", sans-serif;'>VS</h1>", unsafe_allow_html=True)
    with flag_col2:
        st.markdown(f"<div style='text-align: left;'><img src='{flags[bowling_team]}' width='140' style='border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);'/></div>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Match Situation Inputs
    st.markdown("<h3 class='subheader'>Current Match Situation</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_score = st.number_input('Current Score', min_value=0, max_value=350, value=75, step=1)
    with col2:
        overs = st.number_input('Overs Done', min_value=5, max_value=19, value=8, step=1)
    with col3:
        wickets = st.number_input('Wickets Out', min_value=0, max_value=9, value=2, step=1)
    with col4:
        last_five = st.number_input('Runs in Last 5 Overs', min_value=0, max_value=100, value=45, step=1)

    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs if overs > 0 else 0
    aggression = crr * wickets_left
    pressure = wickets_left / balls_left if balls_left > 0 else 0
    progress = overs / 20
    runs_possible = (balls_left/6) * crr

    if overs <= 6:
        phase = "powerplay"
    elif overs <= 15:
        phase = "middle"
    else:
        phase = "death"


    # --- 8. Prediction Execution ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Using columns to push the button to the right side
    _, col_btn = st.columns([3, 1]) 
    with col_btn:
        predict_button = st.button("Predict Final Score", use_container_width=True, type="primary")

    if predict_button:
        input_df = pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':[city],
            'current_score':[current_score],
            'balls_left':[balls_left],
            'wickets_left':[wickets_left],
            'crr':[crr],
            'last_five':[last_five],
            'aggression':[aggression],
            'pressure':[pressure],
            'phase':[phase],
            'progress':[progress],
            'runs_possible':[runs_possible]
        })

        # THE MATH FIX: The model predicts remaining runs, so we add it to the current score.
        result = pipe.predict(input_df)
        predicted_remaining_runs = int(result[0])
        
        # Ensure we don't get weird negative predictions dropping the score
        if predicted_remaining_runs < 0:
            predicted_remaining_runs = 0
            
        predicted_score = current_score + predicted_remaining_runs

        # Calculate a realistic range
        lower = predicted_score - 12
        upper = predicted_score + 12

        st.markdown("---")
        st.markdown("<h3 class='subheader'>Prediction Results</h3>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(f"""
            <div class='metric-container'>
                <h4>Projected Final Score</h4>
                <h1 style='color: #28a745; font-size: 60px; margin: 0;'>{predicted_score}</h1>
            </div>
            """, unsafe_allow_html=True)
        with res_col2:
            st.markdown(f"""
            <div class='metric-container'>
                <h4>Expected Range</h4>
                <h1 style='color: #17a2b8; font-size: 60px; margin: 0;'>{lower} - {upper}</h1>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Chart
        chart_data = pd.DataFrame({
            'Stage': ['Current Score', 'Projected Final Score'],
            'Runs': [current_score, predicted_score]
        })
        
        fig = px.bar(chart_data, x='Stage', y='Runs', text='Runs', color='Stage', 
                     color_discrete_sequence=['#ffc107', '#28a745'], height=400)
        
        fig.update_layout(
            showlegend=False, 
            xaxis_title=None, 
            bargap=0.4,       
            margin=dict(t=30, b=30, l=10, r=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Helvetica Neue", size=14)
        )
        fig.update_traces(textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- 9. Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer'>Developed by Vivek Yadav - Lakshya Goyal - Shirsh Ranjan Ghosh | CSE</div>",
    unsafe_allow_html=True
)