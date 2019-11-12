# mlteam-lime-for-tsne
MLTeam: Apply LIME for t-SNE

# Some dev notes:
We can add some note for environment configuration, hyperparameters, tip and tricks here.

### Add submodule to github repo:

*Ref*: https://gist.github.com/gitaarik/8735255

*How to*: I want to add a repo containing all my simple, common code for any python project. This code is at the [py-common repo](https://github.com/vu-minh/py-common).

```bash
# create a folder in current repo to hold the code in `py-common`
mkdir common

# make the `py-common` repo a submodule in this current repo
git submodule add https://github.com/vu-minh/py-common.git common

# keep the code in `common` up-to-date
git submodule update
```
