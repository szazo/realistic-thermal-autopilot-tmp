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

```shell
# Install and switch Python 3.11
pyenv install 3.11
pyenv local 3.11
 
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
```
### 3. Dependencies
Install the project in editable mode:
```shell
python -m pip install --upgrade pip
python -m pip install --editable ".[dev,thermalmodelling]"
```
### 3. Pull Data
 Pull the versioned data using [DVC](https://dvc.org/):

```shell
dvc pull
```

Alternatively, you can pull the snapshot data directly from [Zenodo (https://sandbox.zenodo.org/records/420119)](https://sandbox.zenodo.org/records/420119) using the following command:

```shell
dvc repro --force --single load_snapshot_archive 
```
### 4. Verification
By default, the data contains the output of each pipeline stages for data preparation, training (which is around ~8 hours for the single agent and ~20 hours for the multi-agent training on Intel i9-12900F with Nvidia RTX 3060 12GB), evaluation and statistics. 

To verify the installation, you can reproduce Figure 5 D and E (and their associated statistics) by running the following analysis:

```shell
dvc repro --single --force generate_altitude_achievement_percent_by_distance_analysis
```

Once the process is complete, the results can be found in the following directory: `results/analysis/from_different_distances_using_train_glider/altitude_achievement_percent` folder.
## Overview
The repository workflow is orchestrated by [DVC](https://dvc.org/). Pipeline stages are defined in `dvc.yaml`, while high-level global parameters are managed in `params.yaml`. Detailed parameters are managed by Hydra, config files found under `config` directory.
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
The repository uses  [MLFlow](https://mlflow.org/)  for experiment tracking, which by default runs locally, there is no need for special configuration.
##### Single Agent Training
You can start the single agent training using the following command (it will run only this stage even its dependencies did not change):

```shell
dvc repro --single --force train_single_agent
```

In a separate terminal you can start MLFlow UI to follow the training process:
```shell
source .venv/bin/activate
mlflow server --host 0.0.0.0 --port 5000
```
Open http://localhost:5000/ in a browser and check the latest `bird_train` *run* under `glider` experiment.

After the training complete the trained policy of the single agent training can be used as peer policy for multi agent training by setting `train.multi_agent.peer_policy.run_id` to the **Run ID** of the new single agent training (found under MLFlow UI), then download the policy files using the following stage:
```shell
dvc repro --single --force download_single_agent_model_for_multi_agent_training
```
The downloaded model (parameters + weights) will be stored under: `data/models/single_agent/RUN_ID`
##### Multi Agent Training
You can start the multi agent training using the following command:
```shell
dvc repro --single --force train_multi_agent
```
You can follow the process of the training in MLFlow (http://localhost:5000/) at the latest `multi_agent_train` run.

After the training completes, select the best weight (based on **`checkpoint/student0/mean_minus_std`** metric under MLFlow) from **step 60 onwards** (ignoring earlier epochs due to curriculum learning). Update `params.yaml`:
```yaml
eval:
  policy:
    run_id: <RUN_ID>
    checkpoint_name: weights<STEP>
```
Then download the model for evaluation:
```shell
dvc repro --single --force download_trained_multi_agent_model
```
#### 3. Evaluation Workflows
Evaluations run the trained multi agent policy in different environment conditions. We conduct evaluations under two primary spatial conditions.

| **Condition**           | **Description**                                                                 | **Output Directory**                                       |
| ----------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Different Distances** | Tests ability to _locate_ thermals from different distances from core.          | `results/eval/realistic/.../from_different_distances_.../` |
| **Close Distance**      | Tests _thermalling efficiency_ inside the lift using the storks' wing loadings. | `results/eval/realistic/.../from_close_distance_.../`      |

Associated `dvc.yaml` stages:
- `eval-peer-informed-student-alone-from-different-distances`
- `eval-peer-informed-student-with-birds-from-different-distances`
- `eval-peer-informed-student-alone-using-bird-wing-loadings-from-close-distance`
- `eval-peer-informed-student-with-birds-using-bird-wing-loadings-from-close-distance`

You can run e.g. one of the evaluation stages using:
```shell
dvc repro --single --force eval-peer-informed-student-alone-from-different-distances
```
#### 4. Analysis & Figures
These stages consume evaluation results to produce the statistical insights used in the paper. Results can be found under `results/analysis`:
- **Thermal Localization**: `generate_altitude_achievement_percent_by_distance_analysis` measures how peer information helps the agent locate the thermal.
- **Thermalling Performance**: `generate_close_distance_vertical_velocity_analysis` compares agent vs. bird climb rates inside realistic thermals (agents spawned inside the thermals). `generate_close_distance_episode_success_analysis` generates episode success analysis.
- **Environmental Robustness**: `generate_different_distances_wind_speed_analysis` examines how wind speed impacts success of locating the thermal.
- **Visualization**: `generate_realistic_thermal_trajectory_plots` creates trajectory comparison plot (Figure 5).
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
