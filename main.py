import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def initialize_population(pop_size, num_genes, green_time_min, green_time_max):
    return np.random.randint(green_time_min, green_time_max, (pop_size, num_genes))


def fitness_function(solution, traffic_data, lost_time):
    incoming_flows = traffic_data["IncomingFlow"].values

    green_signal_times = solution
    repeated_green_signal_times = np.tile(green_signal_times, (len(incoming_flows) // len(green_signal_times)) + 1)[:len(incoming_flows)]

    cycle_time = np.sum(green_signal_times) + lost_time * len(green_signal_times)

    delays = incoming_flows / repeated_green_signal_times
    flow_adjustment = incoming_flows / cycle_time

    total_delay = np.sum(delays)
    total_flow_penalty = np.sum(np.abs(flow_adjustment - 1))
    
    fitness = total_delay + total_flow_penalty
    return -fitness


def select_parents(population, fitness_scores, num_parents):
    parents = np.zeros((num_parents, population.shape[1]))
    for i in range(num_parents):
        best_idx = np.argmax(fitness_scores)
        parents[i, :] = population[best_idx, :]
        fitness_scores[best_idx] = -np.inf 
    return parents


def crossover(parents, offspring_size, crossover_prob):
    offspring = np.zeros(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        
        if np.random.rand() < crossover_prob:
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        else:
            offspring[k, :] = parents[parent1_idx, :]
    return offspring


def mutate(offspring, mutation_rate, green_time_min, green_time_max):
    for idx in range(offspring.shape[0]):
        if np.random.rand() < mutation_rate:
            gene_idx = np.random.randint(0, offspring.shape[1])
            random_value = np.random.randint(green_time_min, green_time_max)
            offspring[idx, gene_idx] = random_value
    return offspring


def genetic_algorithm(traffic_data, pop_size=20, num_genes=4, num_generations=100, 
                      mutation_rate=0.1, crossover_prob=0.7, green_time_min=10, 
                      green_time_max=60, lost_time=4):
    num_parents_mating = pop_size // 2
    population = initialize_population(pop_size, num_genes, green_time_min, green_time_max)
    best_fitness_per_generation = []

    for generation in range(num_generations):
        fitness_scores = np.array([fitness_function(sol, traffic_data, lost_time) for sol in population])
        best_fitness_per_generation.append(np.max(fitness_scores))

        if len(best_fitness_per_generation) > 10 and np.all(np.abs(np.diff(best_fitness_per_generation[-10:])) < 0.001):
            print("GA converged due to fitness stability.")
            break

        if np.max(fitness_scores) >= 0:
            print(f"Stopping early at generation {generation + 1}.")
            break

        parents = select_parents(population, fitness_scores, num_parents_mating)
        offspring_size = (pop_size - parents.shape[0], num_genes)
        crossovered_offspring = crossover(parents, offspring_size, crossover_prob)
        mutated_offspring = mutate(crossovered_offspring, mutation_rate, green_time_min, green_time_max)

        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutated_offspring

        print(f"Generation {generation + 1}: Best Fitness Score = {np.max(fitness_scores)}")

    best_idx = np.argmax(fitness_scores)
    optimal_solution = population[best_idx]

    print("\nBest Optimal Solution for the Junction's Green Signal Times:")
    for i in range(len(optimal_solution)): 
        print(f'Lane: {i+1} => {optimal_solution[i]} secs')

    plt.plot(best_fitness_per_generation, label="Best Fitness Score")
    plt.xlabel("Generations")
    plt.ylabel("Fitness Scores")
    plt.title("Fitness Scores over Generations")
    plt.legend()
    plt.show()

    return optimal_solution


def plot_green_signal_time_distribution(solution, title="Green Signal Time Distribution"):
    lanes = [f"Lane {i+1}" for i in range(len(solution))]
    plt.bar(lanes, solution, color='forestgreen')
    plt.xlabel("Traffic lanes")
    plt.ylabel("Green Signal Time (seconds)")
    plt.title(f"{title}")
    plt.show()


def segment_traffic_data(traffic_data):
    traffic_data = traffic_data.copy()
    traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'], format="%d/%m/%Y %H:%M", errors='coerce')
    valid_traffic_data = traffic_data.dropna(subset=['DateTime']).copy()
    valid_traffic_data['Hour'] = valid_traffic_data['DateTime'].dt.hour
    
    peak_hours = valid_traffic_data[valid_traffic_data['Hour'].isin([7, 8, 9, 13, 14, 17, 20, 21, 22])].copy()
    non_peak_hours = valid_traffic_data[~valid_traffic_data['Hour'].isin([7, 8, 9, 13, 14, 17, 20, 21, 22])].copy()
    
    return peak_hours, non_peak_hours


dataset_path = "C:\Users\DELL\Downloads\AI OEL\AI OEL\traffic.csv"
traffic_data = pd.read_csv(dataset_path)

print("Loaded Traffic Data:\n", traffic_data.head())
traffic_data = traffic_data[['DateTime', 'Junction', 'Vehicles']]
traffic_data['IncomingFlow'] = traffic_data['Vehicles']

junction_data = traffic_data[traffic_data['Junction'] == 1]

peak_data, non_peak_data = segment_traffic_data(junction_data)

print(f"\nRunning optimization for Peak Hours")
optimized_peak = genetic_algorithm(
    peak_data,
    pop_size=30,
    num_genes=4,
    num_generations=20,
    mutation_rate=0.01,
    crossover_prob=0.8,
    green_time_min=20,
    green_time_max=80,
    lost_time=3
)
plot_green_signal_time_distribution(optimized_peak, title="Green Signal Time Distribution (Peak Hours)")

print(f"\nRunning optimization for Non-Peak Hours")
optimized_non_peak = genetic_algorithm(
    non_peak_data,
    pop_size=30,
    num_genes=4,
    num_generations=25,
    mutation_rate=0.01,
    crossover_prob=0.8,
    green_time_min=20,
    green_time_max=80,
    lost_time=3
)
plot_green_signal_time_distribution(optimized_non_peak, title="Green Signal Time Distribution (Non-Peak Hours)")
