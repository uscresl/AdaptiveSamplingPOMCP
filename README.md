# Adaptive Sampling using POMDPs with Domain-Specific Considerations
 
This code is for the paper G. Salhotra*, C. E. Denniston, D. A. Caron and G. S. Sukhatme, "Adaptive Sampling using POMDPs with Domain-Specific Considerations" IEEE International Conference on Robotics and Automation (ICRA), in press


## Steps
Prerequisites: pipenv

1. Clone this repo
2. `cd /path/to/cloned/repo`
3. `pipenv install #it will install from Pipfile`
4. `pipenv shell #enter your new pipenv environment`
5. `python <script_name.py>`


Run the following files to generate experimental results and plot them. Repository does not include the validation environment data so only analytical functions can be recreated.

| File | Description | 
| --- | --- |
| beta_curve_grid_search.py | Run Experiment to generate data for Figure 2 (Comparing different rollout allocator curves) |
| figure_makers/beta_curve_test_plot.py | Plot results from beta_curve_grid_search.py (The best rollout allocation curves) | 
| figure_makers/test_plot_beta_distribution | Plot the beta curves the grid search is tested over. | 
| allocator_test.py | Run experiment to generate data for Figure 3 a b & c, (Compare different allocators) |
| figure_makers/plot_allocator_test.py | Plot the results of allocator_test.py | 
| launch_plan_commit.py | Run experiment to generate data for Figure 3 d & e. (Compare different plan commitment algorithms) |
| figure_makers/plot_plan_commit.py | Plot results of launch_plan_commit.py | 
| launch_combo_expt.py | Run experiment to generate data for Figure 4 (Compare baseline versus all improvements) |
| figure_makers/plot_combo_expt.py | Plot the results of launch_combo_expt.py |
| figure_makers/plot_environment.py | Plot the environments used in the experiments for viewing |

Troubleshooting
* if you have dependency issues when creating the envrionment, try to set `torch="*"` in the Pipfile
* if nothing runs, or if the experiment runner quits immediately, you may have results cached from a previous experiment run. To re-run the experiment, you have to delete (or backup) those results. 

To delete cached results, use either of these options
* Move to trash can (Ubuntu 18.04) 
```
mv experiment_runs/<expt_name>/* ~/.local/share/Trash/files
mv smallab.* ~/.local/share/Trash/files
```

OR:
* Run `rm -rf experiment_runs/<expt_name>/* smallab.*` to _permanently_ delete the results.

### Acknowledgements:
POMCP python implementation from [here](https://github.com/GeorgePik/POMCP).
