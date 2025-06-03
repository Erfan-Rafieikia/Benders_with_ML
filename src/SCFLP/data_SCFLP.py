from dataclasses import dataclass
from config import *  # Import all configuration values
import numpy as np


@dataclass
class Data:
    I: np.ndarray  # Customer index list
    J: np.ndarray  # Facility index list
    S: np.ndarray  # Scenario index list
    demands: np.ndarray  # Customer demands
    scenarios : np.ndarray  # Demand Scenarios
    capacities: np.ndarray  # Facility capacities
    fixed_costs: np.ndarray  # Facility opening costs
    shipment_costs: np.ndarray  # Transportation costs
    max_demand_sum_over_scenario : float  # maximum of demand sums under different scenarios


def word_reader(file_path):
    with open(file_path, "r") as file:
        for line in file:
            for word in line.split():
                yield word

# Scenario generation for demand
def generate_scenarios(demands, num_scenarios=NUM_SCENARIOS, variance_factor=VARIANCE_FACTOR, method="normal", seed=SEED):
    np.random.seed(seed)  # Ensure reproducibility
    scenarios = np.array([
        [max(0, np.random.normal(mu, variance_factor * mu)) for mu in demands]
        for _ in range(num_scenarios)
    ])
    return scenarios


def read_dataset(file_path,num_scenarios=NUM_SCENARIOS, variance_factor=0.2):
    """
    Reads a dataset for the capacitated facility location problem.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        Data: A Data object containing the instance information.
    """
    word = word_reader(file_path)

    # Read the number of facilities and customers
    num_facilities = int(next(word))
    num_customers = int(next(word))

    # Read facility capacities and fixed costs
    capacities = []
    fixed_costs = []
    for _ in range(num_facilities):
        capacity = int(next(word))
        fixed_cost = int(next(word))
        capacities.append(capacity)
        fixed_costs.append(fixed_cost)

    demands = np.array([float(next(word)) for _ in range(num_customers)])

    # Read transportation costs as an m x n matrix
    shipment_costs = np.array(
        [float(next(word)) for _ in range(num_facilities * num_customers)]
    ).reshape(num_facilities, num_customers) #The list is reshaped into a 2D matrix of dimensions (num_facilities, num_customers).

    shipment_costs = np.transpose(shipment_costs)

    # Convert data to numpy arrays
    I = np.arange(num_customers)  # Customer indices, NumPy array with a sequence of numbers starting from 0 up to (but not including) num_customers. For example,[0 1 2] for num_customers=3
    J = np.arange(num_facilities)  # Facility indices
    S = np.arange(num_scenarios)  # Scenario indices
    capacities = np.array(capacities)
    fixed_costs = np.array(fixed_costs)

    # Generate demand scenarios
    scenarios = generate_scenarios(demands, num_scenarios, variance_factor)
    max_demand_sum_over_scenario = max(sum(scenarios[s, i] for i in I) for s in S)
    

    # Print the loaded data for verification
    print(f"Customer indices (I): {I}")
    print(f"Facility indices (J): {J}")
    print(f"Scenario indices (S): {S}")
    print(f"Customer demands: {demands}")
    print(f"Facility capacities: {capacities}")
    print(f"Facility fixed costs: {fixed_costs}")
    print(f"Shipment costs matrix:\n{shipment_costs}")
    print(f"Generated Demand Scenarios (shape={scenarios.shape}):\n{scenarios}")
    print(f"Max demand Sum :{max_demand_sum_over_scenario}")

    return Data(
        I=I,
        J=J,
        S=S,
        demands=demands,
        scenarios=scenarios,
        capacities=capacities,
        fixed_costs=fixed_costs,
        shipment_costs=shipment_costs,
        max_demand_sum_over_scenario = max_demand_sum_over_scenario
    )
