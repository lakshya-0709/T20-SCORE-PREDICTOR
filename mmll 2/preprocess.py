import numpy as np
import pandas as pd
from yaml import safe_load
import os
from tqdm import tqdm
import pickle

# 1. Data Extraction & Preprocessing
print("Step 1: Extracting match information from YAML files...")
filenames = []
for file in os.listdir('data'):
    if file.endswith('.yaml'):
        filenames.append(os.path.join('data', file))

final_dfs_list = []
counter = 1

for file in tqdm(filenames):
    with open(file, 'r') as f:
        data = safe_load(f)
        df = pd.json_normalize(data)
        df['match_id'] = counter
        final_dfs_list.append(df)
        counter += 1

final_df = pd.concat(final_dfs_list, ignore_index=True)

# 2. Data Cleaning
print("Step 2: Cleaning and filtering match data...")
# Drop unnecessary outcome/meta columns if they exist
cols_to_drop = [
    'info.outcome.method', 'info.neutral_venue', 'info.match_type_number', 
    'info.outcome.by.runs', 'info.outcome.by.wickets', 
    'meta.data_version', 'meta.created', 'meta.revision'
]
existing_cols_to_drop = [c for c in cols_to_drop if c in final_df.columns]
final_df.drop(columns=existing_cols_to_drop, inplace=True)

# Filter for male matches and 20-over matches
if 'info.gender' in final_df.columns:
    final_df = final_df[final_df['info.gender'] == 'male']
if 'info.overs' in final_df.columns:
    final_df = final_df[final_df['info.overs'] == 20]

# Remove redundant features
redundant_cols = ['info.gender', 'info.match_type', 'info.overs']
existing_redundant = [c for c in redundant_cols if c in final_df.columns]
final_df.drop(columns=existing_redundant, inplace=True)

# Save intermediate pickle
print("Saving intermediate dataset: dataset_01.pkl")
pickle.dump(final_df, open('dataset_01.pkl', 'wb'))

# 3. Ball-by-ball extraction (1st Innings)
print("Step 3: Extracting ball-by-ball data (1st Innings)...")
matches = final_df
all_matches_list = []
count = 1

# List of match indices to skip (from notebook)
# For this mock example, none of these indices will be reached, but kept for parity.
exclude_indices = [75, 108, 150, 180, 268, 360, 443, 458, 584, 748, 982, 1052, 1111, 1226, 1345]

for index, row in matches.iterrows():
    if count in exclude_indices:
        count += 1
        continue
    
    count += 1
    
    # Check if 1st innings data exists
    try:
        innings_data = row['innings']
        # In Cricsheet YAML, innings is a list of dicts: [{'1st innings': {...}}, {'2nd innings': {...}}]
        first_innings = None
        for item in innings_data:
            if '1st innings' in item:
                first_innings = item['1st innings']
                break
        
        if first_innings is None:
            continue

        ball_of_match = []
        batsman = []
        bowler = []
        runs = []
        player_of_dismissed = []
        teams = []
        batting_team = []
        match_id_list = []
        city = []
        venue = []

        for ball_info in first_innings['deliveries']:
            for ball_key, details in ball_info.items():
                match_id_list.append(count)
                batting_team.append(first_innings['team'])
                teams.append(row['info.teams'])
                ball_of_match.append(ball_key)
                batsman.append(details['batsman'])
                bowler.append(details['bowler'])
                runs.append(details['runs']['total'])
                city.append(row.get('info.city', np.nan))
                venue.append(row['info.venue'])
                
                # Handle wickets
                if 'wicket' in details:
                    player_of_dismissed.append(details['wicket']['player_out'])
                else:
                    player_of_dismissed.append('0')

        loop_df = pd.DataFrame({
            'match_id': match_id_list,
            'teams': teams,
            'batting_team': batting_team,
            'ball': ball_of_match,
            'batsman': batsman,
            'bowler': bowler,
            'runs': runs,
            'player_dismissed': player_of_dismissed,
            'city': city,
            'venue': venue
        })
        all_matches_list.append(loop_df)
    except Exception as e:
        print(f"Skipping match at index {index} due to error: {e}")
        continue

if all_matches_list:
    delivery_df = pd.concat(all_matches_list, ignore_index=True)
else:
    delivery_df = pd.DataFrame()

# Save Dataset_02
print("Saving intermediate delivery data: Dataset_02.csv")
delivery_df.to_csv('Dataset_02.csv', index=False)

# 4. Feature Engineering & Final Filtering
print("Step 4: Performing feature engineering and final filtering...")

def get_bowling_team(row):
    for team in row['teams']:
        if team != row['batting_team']:
            return team
    return np.nan

if not delivery_df.empty:
    delivery_df['bowling_team'] = delivery_df.apply(get_bowling_team, axis=1)
    delivery_df.drop(columns=['teams'], inplace=True)

    # Filtering for top 10 teams
    top_teams = [
        'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 
        'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
    ]
    
    output = delivery_df[
        (delivery_df['batting_team'].isin(top_teams)) & 
        (delivery_df['bowling_team'].isin(top_teams))
    ]

    final_output = output[['match_id', 'batting_team', 'bowling_team', 'ball', 'runs', 'player_dismissed', 'city', 'venue']]

    # Save Dataset_03
    print("Saving final dataset: Dataset_03.csv")
    final_output.to_csv('Dataset_03.csv', index=False)
    print("Done! Both Dataset_02.csv and Dataset_03.csv have been generated.")
else:
    print("Error: No data extracted. Check your YAML files.")
