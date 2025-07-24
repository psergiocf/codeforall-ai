# Setup

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the packages defined on the 'requirements.txt' in the virtual environment
pip install -r [path_to_requirements_file]
```

To check if the virtual environment is properly setup, check where the packages are installed:

```bash
> pip list        # shows only packages in this environment
> which python    # shows virtual environment Python path
```

### 👨🏽‍🍳 Talk with the head chef bot

```bash
> pip install -r src/requirements.txt
> python src/head_chef.py
```

### 🧪 Play with the examples

```bash
> pip install -r examples/requirements.txt
> python [path_to_your_python_file]    # e.g. python examples/prompting.py
```

When you've finished, deactivate the virtual environment

```bash
deactivate
```
