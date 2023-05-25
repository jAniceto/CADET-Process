import os


import warnings
from pathlib import Path


from scipy import optimize
from scipy.optimize import OptimizeWarning
import numpy as np

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import (
    Bool, Switch, UnsignedInteger, UnsignedFloat
)
from CADETProcess.optimization import OptimizerBase
from CADETProcess.reference import ReferenceIO
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Inlet, Outlet, LumpedRateModelWithPores
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process
from CADETProcess.simulator import Cadet
from CADETProcess.optimization import OptimizationProblem
from CADETProcess.comparison import Comparator


from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render


# can be commented out when the debugger launches inside CADET-process
# ROOT_DIR = os.path.join(os.getcwd(),"repos/CADET-Process/")
DATA_DIR = "examples/characterize_chromatographic_system"

def load_data():

    data = np.loadtxt(os.path.join(
        DATA_DIR, "experimental_data/non_pore_penetrating_tracer.csv"), 
        delimiter=',')

    time_experiment = data[:, 0]
    c_experiment = data[:, 1]

    return time_experiment, c_experiment


def create_process():


    component_system = ComponentSystem(['Non-penetrating Tracer'])


    feed = Inlet(component_system, name='feed')
    feed.c = [0.0005]

    eluent = Inlet(component_system, name='eluent')
    eluent.c = [0]

    column = LumpedRateModelWithPores(component_system, name='column')

    column.length = 0.1
    column.diameter = 0.0077
    column.particle_radius = 34e-6

    column.axial_dispersion = 1e-8
    column.bed_porosity = 0.3
    column.particle_porosity = 0.8
    column.film_diffusion = [0]

    outlet = Outlet(component_system, name='outlet')

    flow_sheet = FlowSheet(component_system)

    flow_sheet.add_unit(feed)
    flow_sheet.add_unit(eluent)
    flow_sheet.add_unit(column)
    flow_sheet.add_unit(outlet)

    flow_sheet.add_connection(feed, column)
    flow_sheet.add_connection(eluent, column)
    flow_sheet.add_connection(column, outlet)

    Q_ml_min = 0.5  # ml/min
    Q_m3_s = Q_ml_min/(60*1e6)
    V_tracer = 50e-9  # m3

    process = Process(flow_sheet, 'Tracer')
    process.cycle_time = 15*60

    process.add_event(
        'feed_on',
        'flow_sheet.feed.flow_rate',
        Q_m3_s, 0
    )
    process.add_event(
        'feed_off',
        'flow_sheet.feed.flow_rate',
        0,
        V_tracer/Q_m3_s
    )

    process.add_event(
        'feed_water_on',
        'flow_sheet.eluent.flow_rate',
        Q_m3_s,
        V_tracer/Q_m3_s
    )

    process.add_event(
        'eluent_off',
        'flow_sheet.eluent.flow_rate',
        0,
        process.cycle_time
    )

    return process


# get data
time_experiment, c_experiment = load_data()
tracer_peak = ReferenceIO(
    'Tracer Peak', time_experiment, c_experiment
)


# define process
process = create_process()


# set up simulator
simulator = Cadet()

simulation_results = simulator.simulate(process)
_ = simulation_results.solution.outlet.inlet.plot()


# compare results
comparator = Comparator()
comparator.add_reference(tracer_peak)
comparator.add_difference_metric(
    'NRMSE', tracer_peak, 'outlet.outlet',
)


# plots
# _ = tracer_peak.plot()
# comparator.plot_comparison(simulation_results)


# create optimization problem
optimization_problem = OptimizationProblem('bed_porosity_axial_dispersion')

optimization_problem.add_evaluation_object(process)

optimization_problem.add_variable(
    name='bed_porosity', parameter_path='flow_sheet.column.bed_porosity',
    lb=0.1, ub=0.6,
    transform='auto'
)

optimization_problem.add_variable(
    name='axial_dispersion', parameter_path='flow_sheet.column.axial_dispersion',
    lb=1e-10, ub=0.1,
    transform='auto'
)

optimization_problem.add_evaluator(simulator)

optimization_problem.add_objective(
    comparator,
    n_objectives=comparator.n_metrics,
    requires=[simulator]
)

def callback(simulation_results, individual, evaluation_object, callbacks_dir='./'):
    comparator.plot_comparison(
        simulation_results,
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
        show=False
    )


optimization_problem.add_callback(callback, requires=[simulator])

def optimization_script(optimization_problem):

    ax_client = AxClient()
    ax_client.create_experiment(
        name="non-penetrating Tracer",
        parameters=[
            {
                "name": "bed_porosity",
                "type": "range",
                "bounds": [0.1, 0.6],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                "log_scale": False,  # Optional, defaults to False.
            },
            {
                "name": "axial_dispersion",
                "type": "range",
                "bounds": [1e-10, 1e-1],
                "log_scale": True
            },
        ],
        # TODO: this can be passed from the comparator. I need to check
        #       whether the comparator can be passed or if it needs the
        #       difference metrics themselves.
        # TODO: [Q] is the goal of all difference metrics minimization?
        # this could be done like:
        # {metric: ObjectiveProperties(minimize=True) for metric in 
        #  optimization_problem.objective_labels}
        objectives={"NRMSE": ObjectiveProperties(minimize=True)},
        # parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
        # outcome_constraints=["l2norm <= 1.25"],  # Optional.
    )


    def objective_function(parameters):
        # TODO: This assumes that all parameters are specified in the correct
        # order. This could be made more robust by using a mapping
        x = [value for _, value in parameters.items()]

        y = optimization_problem.evaluate_objectives(x)[0]
        # which standard error is appropriate? 1.0 was assumed as a starting
        # point
        se = 1.0
        return {"NRMSE": (y, se)}


    for i in range(15):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external
        # system.
        ax_client.complete_trial(
            trial_index=trial_index, 
            raw_data=objective_function(parameters))



    ax_client.get_max_parallelism()

    ax_client.generation_strategy.trials_as_df



    best_parameters, values = ax_client.get_best_parameters()
    best_parameters

    render(ax_client.get_contour_plot())

    return ax_client


ax_client = optimization_script(optimization_problem=optimization_problem)





# this is where I start to create my interface
# the goal is now to read out information from the optimization problem and
# pass it to the AxClient
class AxInterface(OptimizerBase):
    """Wrapper around Ax's bayesian optimization API
    
    populate that class step by step
    """

    def run(self):
        pass


# TODO: implement termination criteria
# TODO: implement function plots (callbacks)
# TODO: [Q] how do I run an individual simulation with updated parameters?