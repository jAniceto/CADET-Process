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

root_dir = Path('../../').resolve()
sys.path.append(root_dir.as_posix())

from CADETProcess.reference import ReferenceIO
from CADETProcess.processModel import ComponentSystem, Inlet, Outlet, LumpedRateModelWithPores, FlowSheet, Process
from CADETProcess.simulator import Cadet
from CADETProcess.comparison import Comparator
from CADETProcess.optimization import OptimizationProblem

if __name__ == '__main__':
    def load_example_data(filename):
        data = np.loadtxt(filename, delimiter=',')

        time_experiment = data[:, 0]
        c_experiment = data[:, 1]

        name = os.path.split(filename)[-1].split(".")[0]
        tracer_peak = ReferenceIO(
            name, time_experiment, c_experiment
        )
        return tracer_peak


    # tracer_peak1 = load_example_data('experimental_data/non_pore_penetrating_tracer.csv')
    # tracer_peak2 = load_example_data('experimental_data/non_pore_penetrating_tracer.csv')
    tracer_peak2 = load_example_data('experimental_data/pore_penetrating_tracer.csv')

    # tracer_peak1.plot()
    # tracer_peak2.plot()
    # import matplotlib.pyplot as plt
    # plt.show()

    def create_process(i):
        component_system = ComponentSystem(['Non-penetrating Tracer'])

        feed = Inlet(component_system, name='feed')
        feed.c = [0.0005]

        eluent = Inlet(component_system, name='eluent')
        eluent.c = [0]

        column = LumpedRateModelWithPores(component_system, name='column')

        column.length = 0.1
        column.diameter = 0.0077
        column.particle_radius = 34e-6

        column.axial_dispersion = 1.97610293e-07
        column.bed_porosity = 4.15790100e-01
        column.particle_porosity = 0.8
        column.film_diffusion = [1e-5]

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
        Q_m3_s = Q_ml_min / (60 * 1e6)
        V_tracer = 50e-9  # m3

        process = Process(flow_sheet, f'Tracer_{i}')
        process.cycle_time = 15 * 60

        process.add_event(
            'feed_on',
            'flow_sheet.feed.flow_rate',
            Q_m3_s, 0
        )
        process.add_event(
            'feed_off',
            'flow_sheet.feed.flow_rate',
            0,
            V_tracer / Q_m3_s
        )

        process.add_event(
            'feed_water_on',
            'flow_sheet.eluent.flow_rate',
            Q_m3_s,
            V_tracer / Q_m3_s
        )

        process.add_event(
            'eluent_off',
            'flow_sheet.eluent.flow_rate',
            0,
            process.cycle_time
        )
        return process


    # process1 = create_process(1)
    process2 = create_process(2)

    # %% [markdown]
    # ### Simulator

    # %%

    # simulator1 = Cadet()
    simulator2 = Cadet()

    # if __name__ == '__main__':
    #     simulation_results = simulator.simulate(process)
    #     _ = simulation_results.solution.outlet.inlet.plot()

    # %% [markdown]
    # ### Comparator

    # %%

    def create_comparator(target_data, name=None):
        comparator = Comparator(name=name)
        comparator.add_reference(target_data)
        comparator.add_difference_metric(
            "PeakHeightDiverging", target_data, 'outlet.outlet',
        )
        comparator.add_difference_metric(
            "PeakPositionDiverging", target_data, 'outlet.outlet',
        )
        return comparator


    # comparator1 = create_comparator(tracer_peak1, name="non-pore-pen")
    comparator2 = create_comparator(tracer_peak2, name="pore-pen")

    # if __name__ == '__main__':
        # simulation_results = simulator2.simulate(process2)
        # comparator2.plot_comparison(simulation_results)

    # %% [markdown]
    # ### Optimization Problem

    # %%

    optimization_problem = OptimizationProblem('bed_porosity_axial_dispersion')

    # optimization_problem.add_evaluation_object(process1)
    optimization_problem.add_evaluation_object(process2)

    # optimization_problem.add_evaluator(simulator1, name="non-pore-pen")
    optimization_problem.add_evaluator(simulator2, name="pore-pen")

    # optimization_problem.add_variable(
    #     name='bed_porosity', parameter_path='flow_sheet.column.bed_porosity',
    #     lb=0.1, ub=0.6,
    #     transform='auto'
    # )

    # optimization_problem.add_variable(
    #     name='axial_dispersion', parameter_path='flow_sheet.column.axial_dispersion',
    #     lb=1e-10, ub=0.1,
    #     transform='auto'
    # )

    optimization_problem.add_variable(
        name='particle_porosity', parameter_path='flow_sheet.column.particle_porosity',
        lb=0.2, ub=0.9,
        transform='auto',
        evaluation_objects=[process2]
    )

    optimization_problem.add_variable(
        name='film_diffusion', parameter_path='flow_sheet.column.film_diffusion',
        lb=1e-10, ub=0.1,
        transform='auto',
        evaluation_objects=[process2],
        component_index=0
    )

    # optimization_problem.add_objective(
    #     comparator1,
    #     n_objectives=comparator1.n_metrics,
    #     requires=[simulator1]
    # )
    optimization_problem.add_objective(
        comparator2,
        n_objectives=comparator2.n_metrics,
        requires=[simulator2]
    )


    def callback(simulation_results, individual, evaluation_object, callbacks_dir='./callback/'):
        # if "1" in str(evaluation_object):
        #     comparator1.plot_comparison(
        #         simulation_results,
        #         file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
        #         show=False
        #     )
        if "2" in str(evaluation_object):
            comparator2.plot_comparison(
                simulation_results,
                file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
                show=False,
                plot_individual=False
            )


    optimization_problem.add_callback(callback, requires=[simulator2])

    # %% [markdown]
    # ### Optimizer

    # %%
    from CADETProcess.optimization import U_NSGA3

    optimizer = U_NSGA3()
    optimizer.n_max_gen = 15
    # optimizer.n_cores = 12

    print("Starting optimizer")
    optimization_results = optimizer.optimize(
        optimization_problem,
        use_checkpoint=False
    )

# %%
