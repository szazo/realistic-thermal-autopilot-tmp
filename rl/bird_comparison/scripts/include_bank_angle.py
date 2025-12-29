import os
from pathlib import Path
import numpy as np
import pandas as pd


def process_data():

    base_path = Path(os.path.dirname(__file__)) / '../../../'
    data_path = base_path / 'data/bird_comparison'

    processed_path = data_path / 'processed'
    output_path = processed_path / 'stork_trajectories'

    trajectory_base_path = data_path / 'input_stork_data'
    decomposition_base_path = data_path / 'decomposed_extrapolated_data'

    dirs = [
        'b023_0.1', 'b023_1.1', 'b010_0.1', 'b121_0.1', 'b112_0.2', 'b077_0.1',
        'b072_0.1'
    ]

    for dirname in dirs:

        trajectory_file_path = trajectory_base_path / dirname / 'data.csv'
        decomposition_file_path = (
            decomposition_base_path / dirname /
            'decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed/iterations.csv'
        )

        trajectory_df = pd.read_csv(trajectory_file_path)
        decomposition_df = pd.read_csv(decomposition_file_path)

        trajectory_bird_groupby = trajectory_df.groupby('bird_name',
                                                        group_keys=False)
        decomposition_bird_groupby = decomposition_df.groupby('bird_name',
                                                              group_keys=True)

        assert np.all(
            trajectory_bird_groupby.size() == decomposition_bird_groupby.size(
            )), 'number of records should match'

        def process_group(bird_trajectory_df: pd.DataFrame):

            bird_name = bird_trajectory_df.name

            bank_angle = decomposition_bird_groupby.get_group(
                bird_name)['bank_angle']
            assert isinstance(bank_angle, pd.Series)

            assert bird_trajectory_df.shape[0] == bank_angle.shape[0]

            bird_trajectory_df['roll_raw_rad'] = bank_angle

            return bird_trajectory_df

        def interpolate_bank_angle(df: pd.DataFrame):

            df['roll_interp_rad'] = df['roll_raw_rad'].interpolate(
                method='cubic')

            return df

        def fill_missing_bank_angle(df: pd.DataFrame):

            df['roll_rad'] = df['roll_interp_rad'].ffill().bfill()
            df['roll_deg'] = np.rad2deg(df['roll_rad'])

            return df

        with_bank_angle_df = trajectory_bird_groupby.apply(process_group)
        with_bank_angle_df = with_bank_angle_df.groupby(
            'bird_name', group_keys=False).apply(interpolate_bank_angle)
        with_bank_angle_df = with_bank_angle_df.groupby(
            'bird_name', group_keys=False).apply(fill_missing_bank_angle)

        output_trajectory_dir = output_path / dirname
        output_trajectory_dir.mkdir(parents=True, exist_ok=True)
        output_trajectory_path = output_trajectory_dir / 'data.csv'
        with_bank_angle_df.to_csv(output_trajectory_path, index=False)


process_data()
