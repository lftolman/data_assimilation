# Data-Assimilation

## Summary

Exploration of data assimilation methods for conditionally guassian dynamical systems. 


## Installation

To clone this repository run this command, replacing username with your github username:

```bash
git clone git@github.com:username/data_assimilation.git
```

To install as a package run these commands:

```bash 
cd data_assimilation

pip install -e .
```

To import the Kalman-Bucy filter in a notebook, use this syntax:

```bash
from kalman_bucy import kalman_bucy
```

## Pipeline for Editing 

### Versioning
PATCH version automatically increments on merges to main. MINOR and MAJOR versions should be updated manually when new methods are added or major changes are made.

## Current To Do (Replaced Weekly)

 - [ ] revamp codebase
 - [ ] clean up overleaf
 - [ ] nudge $U(t)$ rather than direct replacement, but keep covariance update
 - [ ] explore various estimates for covariance $R(t)$
    - [ ] mean approximation


