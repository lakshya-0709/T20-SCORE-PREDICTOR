import pandas as pd
import numpy as np

# Note: pandas_profiling is deprecated in newer environments (renamed to ydata_profiling).
# If this fails, you can comment these two lines out.
#from pandas_profiling import ProfileReport

print("Step 1: Loading Dataset_03.csv...")
df = pd.read_csv('./Dataset_03.csv')

# --- Exploratory Data Analysis (EDA) ---
print("Generating EDA HTML report...")
#prof = ProfileReport(df)
#prof.to_file(output_file='EDA.html')

print("Initial Null Values Check:")
print(df.isnull().sum())

# --- Feature Engineering & Transformations ---
print("\nStep 2: Imputing missing cities and filtering...")
# In case of all missing city we can use the venue's first letter as it will be the city.
cities = np.where(df['city'].isnull(), df['venue'].str.split().apply(lambda x: x[0]), df['city'])
df['city'] = cities

# We do not need venue as we have city, so we can drop it.
df.drop(columns='venue', inplace=True)

# Filter for cities where at least 5 matches were played (600+ deliveries)
eligible_cities = df['city'].value_counts()[df['city'].value_counts() > 600].index.tolist()
df = df[df['city'].isin(eligible_cities)].copy() # Added .copy() to prevent SettingWithCopyWarning later

print("Step 3: Calculating running metrics (current score, balls, over, etc.)...")
# Current score
df['current_score'] = df.groupby('match_id')['runs'].cumsum()

# Extract over and ball number
df['over'] = df['ball'].apply(lambda x: str(x).split(".")[0])
df['ball_no'] = df['ball'].apply(lambda x: str(x).split(".")[1])

# Calculate total balls bowled
df['balls_bowled'] = (df['over'].astype('int') * 6 + df['ball_no'].astype('int'))

# Calculate balls left (handles extras leading to >120 balls)
df['balls_left'] = 120 - df['balls_bowled']
df['balls_left'] = df['balls_left'].apply(lambda x: 0 if x < 0 else x)

print("Step 4: Calculating wickets left and CRR...")
# Wickets left calculation
w = df['player_dismissed'].apply(lambda x: 0 if x == '0' else 1)
df['player_dismissed'] = w
df['player_dismissed'] = df.groupby('match_id')['player_dismissed'].cumsum()
df['wickets_left'] = 10 - df['player_dismissed']

# Current Run Rate (CRR)
df['crr'] = (df['current_score'] * 6) / df['balls_bowled']

print("Step 5: Calculating runs in the last 5 overs (Rolling sum)...")
groups = df.groupby('match_id')
match_ids = df['match_id'].unique()
last_five = []

for match_id in match_ids:
    last_five.extend(groups.get_group(match_id)['runs'].rolling(window=30).sum().values.tolist())

df['last_five'] = last_five

print("Step 6: Finalizing dataset...")
# Merge total runs back into the main dataframe
final_df = df.groupby('match_id')['runs'].sum().reset_index().merge(df, on='match_id')

# Select final features (including match_id for group-based splitting)
final_df = final_df[['match_id', 'batting_team', 'bowling_team', 'city', 'current_score', 
                     'balls_left', 'wickets_left', 'crr', 'last_five', 'runs_x']]

# Drop NaN values introduced by the rolling window
final_df.dropna(inplace=True)

# Shuffle the records to prevent model bias
final_df = final_df.sample(final_df.shape[0])

# Save to CSV
final_df.to_csv('Dataset_04.csv', index=False)

print("\nDone! Dataset_04.csv has been successfully generated.")
print("Final Dataset Info:")
print(final_df.info())