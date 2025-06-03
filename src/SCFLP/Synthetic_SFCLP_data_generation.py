from dataclasses import dataclass
import numpy as np
from config import *

# Configuration for the capacitated facility location problem with demand scenarios
SEED = SEED  # random seed for reproducibility
DEMANDS = DEMANDS   # the range of customer demands
CAPACITIES = CAPACITIES  # the range of facility capacities
FIXED_COSTS = FIXED_COSTS # the range of facility fixed costs
SHIPMENT_COSTS = SHIPMENT_COSTS  # the range of shipment costs
NUM_SCENARIOS = NUM_SCENARIOS # Number of demand scenarios generated
VARIANCE_FACTOR = VARIANCE_FACTOR  # # Variance factor for scenario generation
NUM_CUSTOMERS = NUM_CUSTOMERS #Number of customers
NUM_FACILITIES = NUM_FACILITIES  # Number of facilities

@dataclass
class Data:
    I: np.ndarray  # customer index list
    J: np.ndarray  # facility index list
    demands: np.ndarray  # shape: (num_scenarios, num_customers)
    capacities: np.ndarray  # facility capacities (integers)
    fixed_costs: np.ndarray  # facility opening costs (floats)
    shipment_costs: np.ndarray  # transshipment costs (floats)
    num_scenarios: int         # number of scenarios
    scenario_dist_type: str    # scenario distribution type
    scenario_variance: float   # scenario variance factor

def generate_scenarios(demands, num_scenarios=NUM_SCENARIOS, variance_factor=VARIANCE_FACTOR, method="normal", seed=SEED):
    np.random.seed(seed)  # Ensure reproducibility
    if method == "normal":
        scenarios = np.array([
            [max(0, np.random.normal(mu, variance_factor * mu)) for mu in demands]
            for _ in range(num_scenarios)
        ])
    elif method == "uniform":
        scenarios = np.array([
            [max(0, np.random.uniform(mu - variance_factor * mu, mu + variance_factor * mu)) for mu in demands]
            for _ in range(num_scenarios)
        ])
    elif method == "poisson":
        scenarios = np.array([
            [max(0, np.random.poisson(mu)) for mu in demands]
            for _ in range(num_scenarios)
        ])
    elif method == "lognormal":
        scenarios = np.array([
            [max(0, np.random.lognormal(np.log(mu + 1), variance_factor)) for mu in demands]
            for _ in range(num_scenarios)
        ])
    else:
        raise NotImplementedError("Supported methods: 'normal', 'uniform', 'poisson', 'lognormal'.")
    return scenarios.round().astype(int)

def generate_random_instance(num_customers = NUM_CUSTOMERS, num_facilities =  NUM_FACILITIES, num_scenarios=NUM_SCENARIOS, variance_factor=VARIANCE_FACTOR, method="normal"):
    """
    Generate a random instance for the capacitated facility location problem with demand scenarios.

    Args:
        num_customers (int): Number of customers
        num_facilities (int): Number of facilities
        num_scenarios (int): Number of demand scenarios
        variance_factor (float): Variance factor for scenario generation
        method (str): Scenario generation method

    Returns:
        Data: A Data object containing the instance information.
    """

    np.random.seed(SEED)

    I = np.arange(num_customers)
    J = np.arange(num_facilities)

    base_demands = np.random.randint(low=DEMANDS[0], high=DEMANDS[1] + 1, size=num_customers)
    demands = generate_scenarios(base_demands, num_scenarios, variance_factor, method, seed=SEED)
    capacities = np.random.randint(low=CAPACITIES[0], high=CAPACITIES[1] + 1, size=num_facilities)
    fixed_costs = np.random.uniform(
        low=FIXED_COSTS[0], high=FIXED_COSTS[1], size=num_facilities
    ).round(2)

    # Create a cost matrix
    shipment_costs = np.random.uniform(
        low=SHIPMENT_COSTS[0], high=SHIPMENT_COSTS[1], size=(num_customers, num_facilities)
    ).round(2)

    return Data(
        I=I,
        J=J,
        demands=demands,  # shape: (num_scenarios, num_customers)
        capacities=capacities,
        fixed_costs=fixed_costs,
        shipment_costs=shipment_costs,
        num_scenarios=num_scenarios,
        scenario_dist_type=method,
        scenario_variance=variance_factor,
    )