from pathlib import Path
import pandas as pd

input_path = Path(
    'results/eval/realistic/peer_informed/from_different_distances_using_train_glider/student_with_birds/GLID-4924/100_episodes/student_with_birds_424.26406871m_424.26406871m.csv'
)

input_bird_trajectories_path = Path(
    'data/bird_comparison/processed/stork_trajectories_as_observation_log/merged_observation_log.csv'
)

output_dir = Path('data/bird_comparison/processed/video')
output_dir.mkdir(exist_ok=True, parents=True)
output_path = output_dir / 'video.csv'

episode = 1
thermal = 'b010'
agent_id = 'student0'

df = pd.read_csv(input_path)
df = df[(df['thermal'] == thermal) & (df['episode'] == episode) &
        (df['agent_id'] == agent_id)]

# this will be the only episode in the file
df['episode'] = 0

bird_df = pd.read_csv(input_bird_trajectories_path)
bird_df = bird_df[(bird_df['thermal'] == thermal)]

bird_df['agent_id'] = bird_df['bird_name']

result_df = pd.concat([df, bird_df], ignore_index=True)
result_df.to_csv(output_path)
