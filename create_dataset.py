import pandas as pd
import random

random.seed(42) 
# if you placed the dataset in a different directory, change the path here
DATA_PATH = 'data/archive/Resume/Resume.csv'

white_female_fn = ['Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah'] 
african_american_female_fn = ['Aisha', 'Ebony', 'Keisha', 'Kenya', 'Latonya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha']
white_male_fn = ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd']
african_american_male_fn = ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone']
white_ln = ['Baker', 'Kelly', 'McCarthy', 'Murphy', 'Murray', 'O’Brien', 'Ryan', 'Sullivan', 'Walsh']
african_american_ln = ['Jackson', 'Jones', 'Robinson', 'Washington', 'Williams']

original_df = pd.read_csv(DATA_PATH, index_col=False)
modified_df = original_df.loc[original_df.index.repeat(4)].reset_index(drop=True)

race_values = ['White', 'African_American', 'White', 'African_American'] * len(original_df)
gender_values = ['Female', 'Female', 'Male', 'Male'] * len(original_df)
# Assign the values to the 'Race' and 'Gender' columns
modified_df['Race'] = race_values
modified_df['Gender'] = gender_values

combinations = [("White", "Female"), ("African_American", "Female"), ("White", "Male"), ("African_American", "Male")]
for job_category in modified_df['Category'].unique():
    for race, gender in combinations:
        filtered_df = modified_df[(modified_df["Race"] == race) & (modified_df["Gender"] == gender) & (modified_df["Category"] == job_category)]

        if race == 'White' and gender == 'Female':
            fn_bank = white_female_fn
            ln_bank = white_ln
        elif race == 'African_American' and gender == 'Female':
            fn_bank = african_american_female_fn
            ln_bank = african_american_ln
        elif race == 'White' and gender == 'Male':
            fn_bank = white_male_fn
            ln_bank = white_ln
        elif race == 'African_American' and gender == 'Male':
            fn_bank = african_american_male_fn
            ln_bank = african_american_ln            
        first_names = random.choices(fn_bank, k=len(filtered_df))
        last_names = random.choices(ln_bank, k=len(filtered_df))
        modified_df.loc[filtered_df.index, 'First_name'] = first_names
        modified_df.loc[filtered_df.index, 'Last_name'] = last_names

        num_rows = len(filtered_df)
        political_orientation = ["Democratic", "Republican"] * (num_rows//2) + random.sample(["Democratic", "Republican"], (num_rows%2)) 
        random.shuffle(political_orientation)
        modified_df.loc[filtered_df.index, "Political_orientation"] = political_orientation

# if you want to try other categories, you can change the list below
final_df = modified_df.loc[modified_df['Category'].isin(['INFORMATION-TECHNOLOGY', 'TEACHER', 'CONSTRUCTION'])]
# save the dataset
final_df.to_csv('./data/selected_cats_resumes.csv')
