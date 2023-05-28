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


def create_singlecomponent_SMA_LWE(name: str = None, gradient_cv_length: float = None, ):
    # Component System
    component_system = ComponentSystem()
    component_system.add_component('Salt')
    component_system.add_component("Protein X")

    # Binding Model
    binding_model = StericMassAction(component_system, name=name)
    binding_model.is_kinetic = True
    binding_model.adsorption_rate = [0.0, 0.1]
    binding_model.desorption_rate = [0.0, 1000]
    binding_model.characteristic_charge = [0.0, 50]
    binding_model.steric_factor = [0.0, 11.83]
    binding_model.capacity = 1200.0
    # binding_model.reference_liquid_phase_conc = 500
    # binding_model.reference_solid_phase_conc = 1200

    # Unit Operations
    inlet = Inlet(component_system, name='inlet')
    inlet.flow_rate = 6.683738370512285e-8

    column = LumpedRateModelWithPores(component_system, name='column')
    column.binding_model = binding_model

    column.length = 0.014
    column.diameter = 0.02
    column.bed_porosity = 0.37
    column.particle_radius = 4.5e-5
    column.particle_porosity = 0.75
    column.axial_dispersion = 5.75e-8
    column.film_diffusion = column.n_comp * [6.9e-6]

    column.c = [50, 0]
    column.cp = [50, 0]
    column.q = [binding_model.capacity, 0]

    outlet = Outlet(component_system, name='outlet')

    # Flow Sheet
    flow_sheet = FlowSheet(component_system)

    flow_sheet.add_unit(inlet)
    flow_sheet.add_unit(column)
    flow_sheet.add_unit(outlet, product_outlet=True)

    flow_sheet.add_connection(inlet, column)
    flow_sheet.add_connection(column, outlet)

    # Process
    process = Process(flow_sheet, name=name)

    load_duration = 9
    t_gradient_start = 90.0
    gradient_post_wash = 180.0

    if gradient_cv_length is None:
        process.cycle_time = 2000.0
        gradient_duration = process.cycle_time - t_gradient_start - gradient_post_wash
        t_gradient_end = gradient_duration + t_gradient_start
    else:
        gradient_duration = gradient_cv_length * column.volume / inlet.flow_rate[0]
        t_gradient_end = gradient_duration + t_gradient_start
        process.cycle_time = gradient_duration + t_gradient_start + gradient_post_wash

    c_load = np.array([50.0, 1.0])
    c_wash = np.array([50.0, 0.0])
    c_elute = np.array([500.0, 0.0])
    gradient_slope = (c_elute - c_wash) / gradient_duration
    c_gradient_poly = np.array(list(zip(c_wash, gradient_slope)))

    process.add_event('load', 'flow_sheet.inlet.c', c_load)
    process.add_event('wash', 'flow_sheet.inlet.c', c_wash, load_duration)
    process.add_event('grad_start', 'flow_sheet.inlet.c', c_gradient_poly, t_gradient_start)
    process.add_event('grad_end', 'flow_sheet.inlet.c', c_elute, t_gradient_end)
    return process


def create_comparator(target_data, name=None):
    comparator = Comparator(name=name)
    comparator.add_reference(target_data)
    comparator.add_difference_metric(
        "PeakHeight", target_data, 'outlet.outlet', components=["Protein X"]
    )
    comparator.add_difference_metric(
        "PeakPosition", target_data, 'outlet.outlet', components=["Protein X"]
    )
    return comparator


def main():
    optimization_problem = OptimizationProblem('bed_porosity_axial_dispersion')

    comparator_dict = {}
    for cv in [5, 120]:
        experimental_data = load_example_data(f'../load_wash_elute/experimental_data/gradient_elution_{cv} cv.csv')
        process = create_singlecomponent_SMA_LWE(name=f"{cv}cv", gradient_cv_length=cv)
        simulator = Cadet()

        simulator.time_integrator_parameters.init_step_size = 0
        comparator = comparator_dict[f"{cv}cv"] = create_comparator(experimental_data, name=f"{cv}cv")
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

    def callback(simulation_results, individual, evaluation_object, callbacks_dir='./callback/'):
        sec = SecondaryAxis()
        sec.components = ['Salt']
        sec.y_label = '$c_{salt}$'

        comparator_dict[str(evaluation_object)].plot_comparison(
            simulation_results,
            file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
            show=False,
            secondary_axis=sec
        )

    optimization_problem.add_callback(callback, requires=[simulator])

    from CADETProcess.optimization import U_NSGA3

    optimizer = U_NSGA3()
    optimizer.n_max_gen = 30
    # optimizer.n_cores = 12

    print("Starting optimizer")
    optimization_results = optimizer.optimize(
        optimization_problem,
        use_checkpoint=False
    )


if __name__ == '__main__':
    main()
