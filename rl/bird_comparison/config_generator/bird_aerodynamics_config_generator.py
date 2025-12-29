from .api import (Config, AerodynamicsInfo)


class BirdAerodynamicsConfigGenerator:

    def generate(self, bird_aerodynamics: dict[str, AerodynamicsInfo]):

        base = {'defaults': ['/base_simple_aerodynamics', '_self_']}

        configs: list[Config] = []
        for name, aerodynamics in bird_aerodynamics.items():

            bird_params = {
                'CL': aerodynamics.CL,
                'CD': aerodynamics.CD,
                'mass_kg': aerodynamics.mass_kg,
                'wing_area_m2': aerodynamics.wing_area_m2
            }

            config = Config(name=name.lower(), config={**base, **bird_params})
            configs.append(config)

        return configs
