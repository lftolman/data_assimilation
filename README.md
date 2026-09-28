# Data-Assimilation

## Summary

Exploration of data assimilation methods for conditionally Gaussian dynamical systems. 


## Installation

To clone this repository:

```bash
git clone git@github.com:lftolman/data_assimilation.git
```

To install as a package run these commands:

```bash 
cd data_assimilation

pip install -e .
```

To import a filter in a notebook, use this syntax:

```python
from da.filters import CGKF
```

## Pipeline for Editing 

### Workflow

Do any sandbox work in the untracked \scratch directory then clean and put in a visible exploration notebook. When there is a new method, experiment, or other important code, add it to the package and show its implementation in a validation notebook.

`/scratch` → `/notebooks/[filter]/exploration` → `/notebooks/[filter]/validation` & `/src`

### Versioning
PATCH version automatically increments on merges to main. MINOR and MAJOR versions should be updated manually when new methods are added or major changes are made.

## Current To Do (Replaced Weekly)

 - [ ] revamp codebase
 - [ ] clean up overleaf
 - [ ] nudge $U(t)$ rather than direct replacement, but keep covariance update
 - [ ] explore various estimates for covariance $R(t)$
    - [ ] mean approximation


