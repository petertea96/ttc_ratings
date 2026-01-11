## Setup
It is recommended to use `uv` to set up your python environment, see https://confluence.dkcorpit.com/display/DSs/UV, then simply:

```
uv sync
```

When you open this repo, activate the .venv: 
> source .venv/bin/activate


## How to get streamlit installed?

brew install pipx
pipx install streamlit
pipx ensurepath


Verify that streamlit has been downloaded:
> streamlit hello


End a stream lit app by typing <CTRL + C> in the terminal

Run
> streamlit run ./Homepage.py


### Miscellaneous notes
- I don't know why, but I can't use pip or pip3 to install packages. Instead, I use brew install <pkg name>. Running uv sync also does not seem to install the packages for me...

- List all python packages installed
> uv pip list

## Run end to end
1. Scrape skycourt data?
2. Launch streamlit application
