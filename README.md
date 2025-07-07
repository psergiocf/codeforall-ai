# Setup

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the packages defined on the 'requirements.txt' in the virtual environment
pip install -r [path_to_requirements_file]

```

# To run any of the examples

pip install -r examples/requirements.txt

# To run the main project dependencies

pip install -r src/requirements.txt

```

# To check if the virtual environment is properly setup, check where the packages are installed
pip list      # shows only packages in this environment
which python  # shows virtual environment Python path

# When you've finished, deactivate the virtual environment
deactivate
```

# How to run it?

```bash
python [path_to_your_python_file]
```
