# ---
# jupyter:
#   jupytext:
#     formats: md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
import os.path
# %% [markdown]
# # Fit Binding Model Parameters


# %% tags=["remove-cell"]
from pathlib import Path
import sys

import numpy as np

from CADETProcess.plotting import SecondaryAxis
from examples.load_wash_elute.breakthrough_concentration import create_singlecomponent_SMA_bt
from examples.load_wash_elute.lwe_concentration import create_singlecomponent_SMA_LWE

root_dir = Path('../../../../').resolve()
sys.path.append(root_dir.as_posix())

from CADETProcess.reference import ReferenceIO
from CADETProcess.processModel import ComponentSystem, Inlet, Outlet, LumpedRateModelWithPores, FlowSheet, Process, \
    StericMassAction, GeneralRateModel
from CADETProcess.simulator import Cadet
from CADETProcess.comparison import Comparator
from CADETProcess.optimization import OptimizationProblem


def load_example_data(filename):
    data = np.loadtxt(filename, delimiter=',')

    time_experiment = data[:, 0]
    c_experiment = data[:, 1]

    name = os.path.split(filename)[-1].split(".")[0]
    tracer_peak = ReferenceIO(
        name, time_experiment, c_experiment
    )
    return tracer_peak


def create_comparator_for_gradients(target_data, name=None):
    comparator = Comparator(name=name)
    comparator.add_reference(target_data)
    comparator.add_difference_metric(
        "PeakHeight", target_data, 'outlet.outlet', components=["Protein X"]
    )
    comparator.add_difference_metric(
        "PeakPosition", target_data, 'outlet.outlet', components=["Protein X"]
    )
    return comparator


def create_comparator_for_bt(target_data, name=None):
    comparator = Comparator(name=name)
    comparator.add_reference(target_data)
    comparator.add_difference_metric(
        "NRMSE", target_data, 'outlet.outlet', components=["Protein X"]
    )
    return comparator


def create_optimization_problem(experiment_base="gradients"):
    optimization_problem = OptimizationProblem('SMA_binding_parameters')

    comparator_dict = {}
    if experiment_base == "gradients":
        cvs = [5, 30, 120]
    elif experiment_base == "bt":
        cvs = [30, "bt"]
    else:
        raise ValueError(
            f"Experiment_base of {experiment_base} is not one of the supported values: 'gradients' or 'bt'.")

    for cv in cvs:
        if cv == "bt":
            experimental_data = load_example_data(f'../load_wash_elute/experimental_data/breakthrough.csv')
            process = create_singlecomponent_SMA_bt()
            comparator = comparator_dict[f"{cv}cv"] = create_comparator_for_bt(experimental_data, name=f"{cv}cv")
        else:
            experimental_data = load_example_data(f'../load_wash_elute/experimental_data/gradient_elution_{cv} cv.csv')
            process = create_singlecomponent_SMA_LWE(name=f"{cv}cv", gradient_cv_length=cv)
            comparator = comparator_dict[f"{cv}cv"] = create_comparator_for_gradients(experimental_data, name=f"{cv}cv")

        simulator = Cadet()

        simulator.time_integrator_parameters.init_step_size = 0
        optimization_problem.add_evaluation_object(process)
        optimization_problem.add_evaluator(simulator, name=f"{cv}cv")
        optimization_problem.add_objective(
            comparator,
            n_objectives=comparator.n_metrics,
            requires=[simulator],
            evaluation_objects=[process]
        )

        # print("Simulating")
        # simulation_results = simulator.simulate(process)
        #
        # sec = SecondaryAxis()
        # sec.components = ['Salt']
        # sec.y_label = '$c_{salt}$'
        #
        # comparator.plot_comparison(simulation_results, secondary_axis=sec)

    optimization_problem.add_variable(
        name='sma_ka', parameter_path='flow_sheet.column.binding_model.adsorption_rate',
        lb=1e-2, ub=1e4,
        transform='auto',
        component_index=1
    )

    optimization_problem.add_variable(
        name='sma_characteristic_charge', parameter_path='flow_sheet.column.binding_model.characteristic_charge',
        lb=1, ub=50,
        transform='auto',
        component_index=1
    )

    if experiment_base == "bt":
        optimization_problem.add_variable(
            name='sma_kd', parameter_path='flow_sheet.column.binding_model.desorption_rate',
            lb=1e-2, ub=1e4,
            transform='auto',
            component_index=1
        )
        optimization_problem.add_variable(
            name='sma_steric_factor', parameter_path='flow_sheet.column.binding_model.steric_factor',
            lb=1, ub=50,
            transform='auto',
            component_index=1
        )

    # def callback(simulation_results, individual, evaluation_object, callbacks_dir='./callback/'):
    #     sec = SecondaryAxis()
    #     sec.components = ['Salt']
    #     sec.y_label = '$c_{salt}$'
    #
    #     comparator_dict[str(evaluation_object)].plot_comparison(
    #         simulation_results,
    #         file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
    #         show=False,
    #         secondary_axis=sec
    #     )
    #
    # optimization_problem.add_callback(callback, requires=[simulator])
    return optimization_problem


if __name__ == '__main__':
    optim_problem = create_optimization_problem("gradients")
    from CADETProcess.optimization import NSGA2

    optimizer = NSGA2()
    optimizer.n_max_gen = 30
    # optimizer.n_cores = 12

    print("Starting optimizer")
    optimization_results = optimizer.optimize(
        optim_problem,
        use_checkpoint=False
    )
