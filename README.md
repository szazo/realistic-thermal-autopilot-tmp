# Developing an autopilot system for thermal soaring and certifying it in the reconstructed realistic thermals
## Setup

### 1. Cloning the Repository

Clone the main repository and initialize the realistic thermal model repository using [gitman](https://gitman.readthedocs.io/en/latest/):
```bash
git clone https://gitlab.com/avisense-autopilot/glider.git gliderl
cd gliderl
pipx run gitman install
```
### 2. Environment Configuration

We recommend using **Python 3.11**. You can manage your versions with `pyenv`:

```bash
# Install and switch Python 3.11
pyenv install 3.11
pyenv local 3.11
 
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
```
### 3. Dependencies
Install the project in editable mode:
```
python -m pip install --upgrade pip
python -m pip install --editable ".[dev,thermalmodelling]"
```
### 3. Pull Data
 Pull the versioned data using [DVC](https://dvc.org/):

```
dvc pull
```

Alternatively, you can pull the snapshot data directly from [https://sandbox.zenodo.org/records/420119](https://sandbox.zenodo.org/records/420119) using the following command:

```
dvc repro --force --single load_snapshot_archive
```
## Overview
The repository workflow is orchestrated by [DVC](https://dvc.org/). Pipeline stages are defined in `dvc.yaml`, while high-level global parameters are managed in `params.yaml`.
### Project Stucture & Data Flow

```bash
.
├── config                                # Hydra experiment configs for training and eval
├── data
│   └── bird_comparison
│       ├── input_stork_data              # RAW: Source stork trajectories
│       ├── wing_loading_stork            # RAW: Wing loadings of stork individuals
│       ├── decomposed_extrapolated_data  # RAW: Reconstructed realistic thermals
│       └── processed                     # Preprocessed data   
├── results
│   ├── eval                              # Evaluation results
│   ├── analysis                          # Stats and visualizations
│   └── videos                            # Video visualizations
├── rl                                    # Core Python modules
│   ├── assets                            # Assets for visualization
│   ├── bird_comparison                   # Analysis code and Hydra configuration generators
│   ├── distributions                     # Probability distribution configurations
│   ├── env                               # Custom Gymnasimum and PettingZoo environments
│   ├── model                             # PyTorch model implementations
│   ├── scripts                           # Script entry points
│   ├── thermal                           # Thermal models (Gaussian and realistic)
│   ├── trainer                           # Training logic (built with Tianshou)
│   ├── utils                             # Utility functions and helper scripts
│   └── visualization                     # 3D visualization
└── thirdparty                            # Realistic thermal model package
```
### Pipeline Stages
#### 1. Data Preparation & Config Generation
These stages transform raw data into a format the simulation environment can use.
- `download_bird_data`: Download stork trajectories and thermal reconstructions
- `preprocess_bird_data`: Augments trajectories with bank angle calculations.
- `generate_bird_experiment_configs`: Creates Hydra configs for stork individuals' aerodynamics and evaluation settings (under `config/birds/realistic/eval`)
- `prepare_bird_trajectories`: Replays bird trajectories in thermal models to generate standardized observation logs.
#### 2. Training
The repository uses [neptune.ai](https://neptune.ai)  for experiment tracking, as weight storage, and logging.
##### 1. Neptune Setup
To log your training runs, you must link your own Neptune workspace:
1. **Project Config:** Open `config/birds/logger/neptune_logger.yaml` and update the `project:` field with your workspace and project name (e.g. `workspace/project-name`).
2. **Authentication:** Export your API token as an environment variable: 
```bash
export NEPTUNE_API_TOKEN="your_token"
```
##### 2. Training Stages

Training is handled via DVC stages. These utilize the parameters defined in `params.yaml`:
- **Single Agent Training:** (`train_single_agent`) Trains a single glider.
- **Multi-Agent Training:** (`train_multi_agent`) Trains an agent in the presence of others to develop peer-informed behaviors.
##### 3. Using trained policies

The `params.yaml` acts as the central hub for model selection. By providing a **Neptune Run ID** (e.g. `GLID-1234`), you can specify which weights to load for different tasks:

| **Purpose**               | **Parameters in params.yaml**              | Notes                                                                |
| ------------------------- | ------------------------------------------ | -------------------------------------------------------------------- |
| **Peer Policy**           | `train.multi_agent.peer_policy.neptune_id` | Selects the best weights based on the reward formula: $mean-std$     |
| **Evaluation**            | `eval.policy.neptune_id`                   | Loads the model for evaluation                                       |
| **Evaluation Checkpoint** | `eval.policy.checkpoint_name`              | (Optional) Allows selection of a specific checkpoint within the run. |
#### 3. Evaluation Workflows
We conduct evaluations under two primary spatial conditions.

| **Condition**           | **Description**                                                                 | **Output Directory**                                       |
| ----------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Different Distances** | Tests ability to _locate_ thermals from different distances from core.          | `results/eval/realistic/.../from_different_distances_.../` |
| **Close Distance**      | Tests _thermalling efficiency_ inside the lift using the storks' wing loadings. | `results/eval/realistic/.../from_close_distance_.../`      |

Associated `dvc.yaml` stages:
- `eval-peer-informed-student-alone-from-different-distances`
- `eval-peer-informed-student-with-birds-from-different-distances`
- `eval-peer-informed-student-alone-using-bird-wing-loadings-from-close-distance`
- `eval-peer-informed-student-with-birds-using-bird-wing-loadings-from-close-distance`
#### 4. Analysis & Figures
These stages consume evaluation results to produce the statistical insights used in the paper. Results can be found under `results/analysis`:
- **Thermal Localization**: `generate_altitude_achievement_percent_by_distance_analysis` measures how peer information helps the agent locate the thermal.
- **Thermalling Performance**: `generate_close_distance_vertical_velocity_analysis` compares agent vs. bird climb rates inside realistic thermals (agents spawned inside the thermals).
- **Environmental Robustness**: `generate_different_distances_wind_speed_analysis` examines how wind speed impacts success of locating the thermal.
- **Visualization**: `generate_realistic_thermal_trajectory_plots` creates trajectory comparison plot.
#### 5. Video Generation
* **Realistic Thermal**: Shows agent behavior when flying in realistic thermal along birds: `generate_glider_with_birds_in_realistic_thermal_video`
* **Training**: Generates videos about the agent during training. `dvc.yaml` stages:
	* `single_agent_eval_for_training_video_160m`
	* `generate_single_agent_training_video_160m`
	* `single_agent_eval_for_training_video_300m`
	* `generate_single_agent_training_video_300m`
### Running the Pipeline
To reproduce a specific analysis (e.g., the performance from different distances), use:

```
dvc repro --single --force generate_altitude_achievement_percent_by_distance_analysis
```
