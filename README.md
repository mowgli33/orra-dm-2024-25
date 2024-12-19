<p align="center">
  <img aligne="center" src="images/CS.png" width="45%" />
  <img aligne="center" src="images/artefact.png" width="50%" ver />
</p>

# Decision Making
Auriau Vincent,
Belahcene Khaled,
Mousseau Vincent

## Table of Contents
- [Repository Usage](#repository-usage)
- [Tasks](#tasks)
- [Deliverables](#deliverables)
- [Resources](#resources)

## Repository usage
1.  Install [git-lfs ](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), it will be needed to download the data
2. Clone the repository preferably using *ssh*
3. Make sure that git-lfs downloaded the files in data/. With the command ```du -sh *```the files use several Mo of memory.

The command 
```bash
conda env create -f config/env.yml
conda activate cs_td
python evaluation.py
``````
will be used for evaluation, with two other test datasets. Make sure that it works well.

## Dataset

You can find the first dataset in [data/dataset_4](./data/dataset_4/). It contains three files: *X.npy*, *Y.npy* and *Z.npy*. They are organised so that $X[i] \succeq_{Z[i]} Y[i]$. Which means that the *i-th* element of X has been preferred to the *i-th* element of Y by the cluster described by the *i-th* element of Z.
Of course Z is provided only for you to check your solution and shouldn't be used for the modelization.
You can also use an equivalent with more features and pairs provided in [data/dataset_10](./data/dataset_10/).

The second dataset needs to be downloaded through the choice-learn package. This [notebook](notebooks/loading_cars_data/ipynb) provides a few indications.

## Tasks
You are asked to:
  - Write a Mixed-Integer Progamming model that would solve both the clustering and learning of a UTA model on each cluster
  - Code this MIP inside the TwoClusterMIP class in python/model.py. It should work on the dataset_4 dataset.
  - Explain and code a heuristic model that can work on the cars dataset. It should be done inside the HeuristicModel class.

## Deliverables
You will present your results during an oral presentation organized the on Tuesday $13^{th}$ (from 1.30 pm) of February. Exact time will be communicated later. Along the presentation, we are waiting for:

-  A report summarizing you results as well as your thought process or even non-working models if you consider it to be interesting.
-  Your solution of the first assignement should be clearly written in this report. For clarity, you should clearly state variables, constraints and objective of the MIP.
-  A well organized git repository with all the Python code of the presented results. A GitHub fork of the repository is preferred. Add some documentation directly in the code or in the report for better understanding. The code must be easily run for testing purposes.
- In particular the repository should contain your solutions in the class TwoClustersMIP and HeuristicModel in the models.py file.  If you use additional libraries, add them inside the config/env.yml file. The command 'python evaluation.py' will be used to check your models, be sure that it works and that your code complies with it. The dataset used will be a new one, with the same standards as 'dataset\_4'.

## Resources
- [Gurobi](https://www.gurobi.com/)
- [Example Jupyter Notebook](notebooks/example.ipynb)
- [UTA model](https://www.sciencedirect.com/science/article/abs/pii/0377221782901552)
