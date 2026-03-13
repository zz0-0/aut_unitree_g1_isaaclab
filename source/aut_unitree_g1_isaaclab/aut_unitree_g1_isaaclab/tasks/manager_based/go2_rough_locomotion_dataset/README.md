# GO2 Quadruped Training Setup

This directory contains the configuration for training the Unitree GO2 quadruped robot on rough terrain locomotion tasks. The setup is designed as a **control group** to compare with the G1 humanoid robot training.

## Key Design Principles

To maintain consistency between GO2 and G1 training for fair comparison:

1. **Same Training Hyperparameters**: Both use identical PPO parameters (learning rate, batch size, etc.)
2. **Same Reward Structure**: Similar reward terms adapted for quadruped vs. humanoid morphology
3. **Same Environment Settings**: Matching terrain generation, randomization, and curriculum settings
4. **Same Command Ranges**: Forward velocity limited to [0.0, 1.0] m/s (no backward motion)

## Differences from G1 (Due to Robot Morphology)

- **Action Scale**: 0.25 for GO2 vs higher for G1 (quadruped requires smaller scale)
- **Contact Bodies**: GO2 uses `.*_foot` for feet air time, G1 uses ankle joints
- **Base Link**: GO2 uses `base`, G1 uses `torso_link`
- **Initial Height**: GO2 at 0.4m, G1 at ~0.75m
- **Reward Weights**: Slightly adjusted for quadruped stability (e.g., feet_air_time: 0.01)

## Files

- `go2_rough_locomotion_data_set_env_cfg.py` - Environment configuration
- `agents/rsl_rl_ppo_cfg.py` - PPO training configuration
- `__init__.py` - Gym environment registration

## Training Commands

### Train GO2
```bash
cd /home/ryz5920/Project/aut_unitree_g1_isaaclab
python scripts/reinforcement_learning/rsl_rl/train.py --task Go2-Rough-Locomotion-Dataset-v0 --num_envs 4096
```

### Play Trained Model
```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task Go2-Rough-Locomotion-Dataset-Play-v0 --num_envs 50
```

## Comparison with G1

| Aspect | GO2 | G1 |
|--------|-----|-----|
| Robot Type | Quadruped | Humanoid |
| DOFs | 12 (3 per leg) | 29 (legs, arms, waist, hands) |
| Base Height | 0.4m | 0.75m |
| Action Scale | 0.25 | 0.5 (default) |
| Actuators | DC Motor | Implicit PD |
| Network Size | 512-256-128 | 512-256-128 |
| Learning Rate | 1e-3 | 1e-3 |
| Training Iterations | 3000 | 3000 |

## Notes

- Both robots use the same terrain generation parameters for rough terrain
- Both disable backward velocity commands (lin_vel_x >= 0.0)
- Both use identical PPO algorithm settings for fair comparison
- The GO2 configuration is based on IsaacLab's validated GO2 setup
- Contact sensors and reward terms are adapted for quadruped morphology
