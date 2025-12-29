from pathlib import Path
import numpy as np
import pandas as pd
from gymnasium.utils.seeding import np_random

from .api import AerodynamicsInfo


class BirdAerodynamicsLoader:

    def load(self, input_filepath: Path, seed: int | None):

        wing_loading_df = pd.read_csv(input_filepath, index_col=0)
        rnd, _ = np_random(seed=seed)

        birds: dict[str, AerodynamicsInfo] = {}

        for index, row in wing_loading_df.iterrows():
            name = str(index)
            wing_loading_kg_per_m2 = row['WL_mode_avg']

            mass_kg = float(
                np.round(rnd.uniform(low=2.5, high=3.5), decimals=2))
            wing_area_m2 = float(1 / (wing_loading_kg_per_m2 / mass_kg))

            assert np.isclose(mass_kg / wing_area_m2, wing_loading_kg_per_m2)

            bird_params = AerodynamicsInfo(CL=1.5,
                                           CD=0.09,
                                           mass_kg=mass_kg,
                                           wing_area_m2=wing_area_m2)
            birds[name] = bird_params

        return birds
