# flare-2022

## Train using CLI

### Create a config

Create a config in `configs` directory similar to ones already there.

### Activate virtual environment

```bash
source .venv/bin/activate
```

### Train the model using the config

```bash
python trainer_cli.py fit --config configs/version1.yaml
```