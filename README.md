# Survey Analysis

## Setup

Optionally, create a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Install the requirements:
```
pip install -r requirements.txt
```

## Using the Notebooks

You can use [Jupytext](https://github.com/mwouts/jupytext) to make Jupyter notebooks friends
with git. Follow Jupytext docs to either install a Jupyter extension and pair the script,
or use manual conversion:

```
jupytext --sync survey_analysis.ipynb
```

Now, you can only commit the `.py` file to the repo.
