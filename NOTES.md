# DEVELOPMENT NOTES
I need several scripts and functions:
1. One that reads a .txt (or better, json) file with a system of differential equations, plus initial conditions, and generates data (see src/createDataset/createDataset.py).
2. One that transforms the dataset by using the 'trick' in the paper, the approximate Eulerian form (see src/createDeltat/createDeltat.py).
3. Run separate instances of pySR on each equation, obtaining several Pareto fronts of candidate solutions.
4. Pair equations found by pySR, converting them back to their original dx/dt format (ODE systems).
5. Identify parameters inside the ODE systems, and optimize them using CMA-ES against the original data.
6. Return the best ODE system.

Currently, functions #1-#4 are complete. The code for function #5 is probably almost ready, but something needs to be re-thought to better pre-select the candidate systems. And also find a way to evaluate the candidates that is more informative that just R2 or MSE.

Ideally, a lot of experiments can be run, changing:
- noise applied to the data
- regular/irregular sampling

## TODO
- Also store the random seeds for each experiment.
- perform a cross-validation, select only equations with recurring structure that appear in the Pareto fronts
- What kind of modified fitness function (based on error) could be used for giving more importance to the beginning of the dynamic? Check the publication "Correlation versus RMSE loss functions in symbolic regression tasks"
- More generally, let's say that I have two candidate systems that are both making errors on a dynamic; which one is the 'best' one? Using an aggregate measure like R2 just 
- Some papers on evaluating similarity between ODE systems: 
	- "Similarity Between Two Stochastic Differential Systems"
	- "A General Metric for the Similarity of Both Stochastic and Deterministic System Dynamics"

## 2024-07-13
There is a problem somewhere in the $f(x, t) -> F_x(x, t, delta_t)$ conversion. The original equation in some cases (e.g. Rossler's F_x_2) does not seem to perform better than others, that are clearly wrong (even wrong structure!). Possible reasons of the issues:
1. There is a bug somewhere in the symbolic process that performs the transformation. But this is kinda weird, because it works in other cases.
2. There is a mathematical issue with the original formula. Maybe something related to how constants gets transformed in the other feature space?

To be explored, because this is an issue for *both* chaotic and non-chaotic systems.