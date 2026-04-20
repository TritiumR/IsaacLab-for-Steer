# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Isaac Lab is a GPU-accelerated robotics simulation framework built on NVIDIA Isaac Sim. It provides a unified interface for reinforcement learning, imitation learning, and motion planning. The framework supports two environment design workflows:

- **Manager-based**: Decomposes environments into configurable managers (observations, actions, rewards, terminations, events, curriculum)
- **Direct**: Single-class implementation where users implement all functionality directly

## Common Commands

All development commands are handled through the `./isaaclab.sh` (Linux) or `isaaclab.bat` (Windows) script.

### Environment Setup

```bash
# Create conda environment (default name: env_isaaclab)
./isaaclab.sh --conda [env_name]

# Create uv environment
./isaaclab.sh --uv [env_name]

# Install Isaac Lab extensions and RL frameworks
./isaaclab.sh --install [all|none|rsl_rl|rl_games|sb3|skrl]
```

### Running Code

```bash
# Run Python with Isaac Sim environment
./isaaclab.sh --python <script.py>

# Launch Isaac Sim GUI
./isaaclab.sh --sim

# Run RL training with RSL-RL
./isaaclab.sh --python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-v0 --headless

# Run RL inference
./isaaclab.sh --python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Lift-Cube-v0 --num_envs 32 --checkpoint <path>

# List all registered environments
./isaaclab.sh --python scripts/environments/list_envs.py
```

### Testing and Linting

```bash
# Run all tests
./isaaclab.sh --test

# Run a specific test file
./isaaclab.sh --python -m pytest tools/template/test/test_generator.py -v

# Run pre-commit formatting and linting
./isaaclab.sh --format

# Run pre-commit on specific files
pre-commit run --files <file1.py> <file2.py>
```

### Documentation

```bash
# Build documentation
./isaaclab.sh --docs

# Open built documentation
xdg-open docs/_build/current/index.html
```

### Docker

```bash
# Build and run Docker container
python3 docker/container.py
```

### Project Templates

```bash
# Create new task or project from template
./isaaclab.sh --new
```

## Architecture

### Package Structure

The repository is organized as a monorepo with multiple Python packages:

- **`source/isaaclab/`**: Core framework providing simulation interface, environments, sensors, assets, and managers
- **`source/isaaclab_assets/`**: Robot and object asset configurations
- **`source/isaaclab_tasks/`**: Task implementations (manager-based and direct workflows)
- **`source/isaaclab_rl/`**: RL framework integrations (RSL-RL, RL-Games, Stable-Baselines3, SKRL)
- **`source/isaaclab_mimic/`**: Imitation learning support (Apache 2.0 licensed)

### Manager-Based Workflow

Manager-based environments use a configuration-driven approach with these key managers:

- **ActionManager**: Applies actions to assets (joint positions, joint velocities, inverse kinematics)
- **ObservationManager**: Computes observations from scene entities
- **RewardManager**: Computes reward signals
- **TerminationManager**: Determines episode termination conditions
- **EventManager**: Handles domain randomization and scene resets
- **CommandManager**: Generates high-level commands for the robot
- **CurriculumManager**: Manages curriculum learning

Configuration classes (e.g., `ObservationTermCfg`, `RewardTermCfg`) specify:
- `func`: The function to call
- `weight`: Weight for rewards/observations
- `params`: Parameters passed to the function

### Environment Configuration Pattern

Environment configurations use a hierarchical `@configclass` pattern:

```python
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
```

Tasks are registered via Gymnasium entry points in `source/isaaclab_tasks/config/extension.toml`.

### Key Directories for Tasks

- **`source/isaaclab_tasks/isaaclab_tasks/manager_based/`**: Manager-based task implementations
  - `manipulation/`, `locomotion/`, `navigation/`, `classic/`
  - Each task has `mdp/` subdirectory with reward/observation/termination functions
  - `config/` subdirectory with environment configurations
- **`source/isaaclab_tasks/isaaclab_tasks/direct/`**: Direct workflow implementations

### Asset Configuration

Assets are configured using dataclasses in `isaaclab.assets`:

- **`ArticulationCfg`**: Robots with joints
- **`RigidObjectCfg`**: Static or dynamic rigid objects
- **`DeformableObjectCfg`**: Deformable objects
- **`AssetBaseCfg`**: Base class for all assets

Spawn configurations control how assets are loaded:
- `UsdFileCfg`: Load from USD file
- `MjcfFileCfg`: Load from MJCF/URDF
- `SphereCfg`, `CapsuleCfg`, `CuboidCfg`: Procedural primitives

### Sensor System

Sensors are configured and attached to scene entities:

- **`CameraCfg`**: RGB/depth/segmentation cameras (RTX-based)
- **`RayCasterCfg`**: Ray-casting for height maps or distance sensors
- **`ContactSensorCfg`**: Contact force/torque sensing
- **`ImuCfg`**: Inertial measurement units
- **`FrameTransformerCfg`**: Transform tracking between frames

### RL Training Scripts

Training scripts follow a consistent pattern across RL frameworks:

1. Parse arguments with framework-specific CLI args
2. Launch Isaac Sim via `AppLauncher`
3. Import and instantiate environment
4. Configure and run training

Each framework has `train.py` and `play.py` in `scripts/reinforcement_learning/<framework>/`.

## Code Style

- **Line length**: 120 characters
- **Formatter**: Black with `--unstable` flag
- **Import sorting**: isort with custom sections (see `pyproject.toml`)
- **Linting**: flake8 with specific ignores (E402, E501, W503, E203, D401, R504, R505, SIM102, SIM117, SIM118)
- **Type checking**: Pyright (basic mode, missing imports ignored in CI)
- **License headers**: BSD-3 for most files, Apache 2.0 for `isaaclab_mimic`

## Important File Patterns

- `**/mdp/*.py`: Markov Decision Process functions (rewards, observations, terminations)
- `**/config/*.py`: Environment and agent configurations
- `**/agents/*.py`: RL algorithm configurations
- `**/*_env_cfg.py`: Environment configuration classes
- `**/*_cfg.py`: General configuration classes

## Environment Variables

- `ISAACLAB_PATH`: Root of the repository
- `ISAAC_PATH`: Isaac Sim installation directory (set by `_isaac_sim/setup_conda_env.sh`)
- `CARB_APP_PATH`, `EXP_PATH`: Isaac Sim internal paths
