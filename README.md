## How to install:
Run the following sequence of all commands
```
make virtulenv
source venv/bin/activate
make all
pre-commit install
```
This should create a virtual env called venv, install all dependencies,
install the package and activate the created virtual environment.

# How to use pre-commit
`pre-commit` will run automatically as a hook so that to commit things need
to adhere to the linter. To make sure this is the case you can use the following
work-stream. Assume there are files `file.txt` and `scripty.py`. Then the workflows is
```
git add file.txt
git add scripty.py
pre-commit
... [fix all of the things that can't be automatically fixed ] ...
git add file.txt
git add script.txt
git commit -m "some message"
```
