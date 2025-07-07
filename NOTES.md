# DEVELOPMENT NOTES
The general idea is:
a. First compare the data transformations on all data sets in ODEBench.
b. If they make different mistakes in different points, use the three data transformations as three different objectives for a multi-objective approach.
c. It would also be nice to compare against a completely explicit system identification for SR approach, with complex genomes and fitness solving for initial conditions + compare against data.

## State of the repository
Unfortunately, now most of the code is inside the `\utils` folder.

## TODO
The code for C_x is now running. So, the next steps are:
a. Run PySR and check whether we find wrong equations with better performance for the "problematic" systems.
b. Check what happens on ground truth equations with noisy data (+ smoothing).

## 2024-11-21
Now, the next steps for the base comparison between different data transformation is:
1. compute data transformations (both F_x and delta_x) with all possible hyperparameter variations
2. check the fitting of the ground-truth solution against the data 

## 2024-07-27
So, a thorough run on ODEBench and some extra experiments show that the issue is related to the size of $\Delta_t$. The same value that works well for some systems works awfully for others; still, reducing $\Delta_t$ always improves the results. I wonder if there are some huge errors only for a few points, and maybe what should be really optimized is the median error, instead of MSE.

## 2024-07-13
There is a problem somewhere in the $f(x, t) -> F_x(x, t, delta_t)$ conversion. The original equation in some cases (e.g. Rossler's F_x_2) does not seem to perform better than others, that are clearly wrong (even wrong structure!). Possible reasons of the issues:
1. There is a bug somewhere in the symbolic process that performs the transformation. But this is kinda weird, because it works in other cases.
2. There is a mathematical issue with the original formula. Maybe something related to how constants gets transformed in the other feature space?

To be explored, because this is an issue for *both* chaotic and non-chaotic systems.

## Old notes
This is a refactoring of the Symbolic Regression for systems of Ordinary Differential Equations project.

The original repository is here: https://github.com/albertotonda/eureqa-differential-equations
