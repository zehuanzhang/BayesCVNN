import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import copy


# Sample accuracy predictor for testing
class AccuracyPredictor:
    def __init__(self, df):
        self.df = df

    def predict_acc(self, layer_settings):
        # This will predict the accuracy based on the row and layer_setting
        accs = []
        for setting in layer_settings:
            row = self.df[self.df['layer_setting'] == setting]
            if row.empty:
                accs.append(0)  # If no match, return 0 accuracy (or any default value)
            else:
                accs.append(row['accuracy'].values[0])
        return np.array(accs)


# Sample ECE predictor for testing
class ECEPredictor:
    def __init__(self, df):
        self.df = df

    def predict_ece(self, layer_settings):
        # This will predict the ECE based on the row and layer_setting
        eces = []
        for setting in layer_settings:
            row = self.df[self.df['layer_setting'] == setting]
            if row.empty:
                eces.append(1)  # If no match, return a default ECE value (1 for max ECE)
            else:
                eces.append(row['ece'].values[0])
        return np.array(eces)


# Function to calculate the number of dropouts in a layer setting
def num_dropout(layer_setting):
    layer_setting = eval(layer_setting) if isinstance(layer_setting, str) else layer_setting
    return sum([1 if x in [1, 2] else 2 for x in layer_setting])


# NumDropoutPredictor class
class NumDropoutPredictor:
    def __init__(self):
        pass

    def predict_num_dropout(self, layer_settings):
        # This will predict the number of dropouts for each layer setting
        return np.array([num_dropout(setting) for setting in layer_settings])


# Sample efficiency predictor (if required, here it's a placeholder)
class EfficiencyPredictor:
    def get_efficiency(self, sample):
        # Assuming we have some constraint on efficiency, for now, return 0
        return 0


# Updated EvolutionFinder class
class EvolutionFinder:
    def __init__(self, efficiency_predictor, accuracy_predictor, ece_predictor, num_dropout_predictor, population_size, max_time_budget, num_dropout_constraint, **kwargs):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.ece_predictor = ece_predictor
        self.num_dropout_predictor = num_dropout_predictor

        # evolution hyper-parameters
        self.arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.8)
        self.population_size = kwargs.get("population_size", population_size)
        self.max_time_budget = kwargs.get("max_time_budget", max_time_budget)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)
        self.num_dropout_constraint = kwargs.get("num_dropout_constraint", num_dropout_constraint)

    # def random_valid_sample(self, possible_settings):
    #     # Randomly sample a layer_setting
    #     return random.choice(possible_settings)

    def random_valid_sample(self, possible_settings):
        # Randomly sample a layer_setting that meets the num_dropout constraint
        valid_settings = [setting for setting in possible_settings if
                          num_dropout(setting) <= self.num_dropout_constraint]

        while True:
            if valid_settings:
                return random.choice(valid_settings)


    def mutate_sample(self, sample, possible_settings):
        # Mutate a layer_setting randomly by picking a different one
        while True:
            if random.random() < self.arch_mutate_prob:
                new_sample = random.choice(possible_settings)
                if num_dropout(new_sample) <= self.num_dropout_constraint and new_sample != sample:
                    return new_sample
                # if new_sample != sample:  # Ensure it’s different
                #     return new_sample

    def crossover_sample(self, sample1, sample2):
        # Convert both samples to lists (assuming they are tuples or lists)
        sample1 = eval(sample1)
        sample2 = eval(sample2)

        while True:
            new_sample = list(copy.deepcopy(sample1))
            sample2 = list(sample2)

            num_elements = len(new_sample)

            # For each element, randomly choose from either sample1 or sample2
            for i in range(num_elements):
                new_sample[i] = random.choice([sample1[i], sample2[i]])

            # Ensure the new sample meets the num_dropout constraint
            if num_dropout(new_sample) <= self.num_dropout_constraint:
                return tuple(new_sample)


    def run_evolution_search(self, possible_settings, verbose=False, **kwargs):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        self.population_size = kwargs.get("population_size", 20)
        self.max_time_budget = kwargs.get("max_time_budget", 5)

        population = []
        for _ in range(self.population_size):
            sample = self.random_valid_sample(possible_settings)
            if sample:
                population.append(sample)
        print('self.random_valid_sample')

        best_valid = -float('inf')  # Initialize to negative infinity
        best_sample = None
        if verbose:
            print("Start Evolution...")

        with tqdm(total=self.max_time_budget, desc="Searching for best layer_setting") as t:
            for i in range(self.max_time_budget):
                # Evaluate accuracy, ECE, and num_dropout of the current population
                accs = self.accuracy_predictor.predict_acc(population)
                eces = self.ece_predictor.predict_ece(population)
                num_dropouts = self.num_dropout_predictor.predict_num_dropout(population)

                # Calculate the objective: accuracy - ece - num_dropout
                # scores = accs# - eces - num_dropouts
                scores = - eces

                # Sort population based on the objective
                sorted_population = [x for _, x in sorted(zip(scores, population), reverse=True)]

                if scores.max() > best_valid:
                    best_valid = scores.max()
                    best_sample = sorted_population[0]

                # Select the best parents
                parents = sorted_population[:int(self.parent_ratio * self.population_size)]

                # Create new population by mutation and crossover
                new_population = []
                for _ in range(int(self.mutation_ratio * self.population_size)):
                    parent = random.choice(parents)
                    new_sample = self.mutate_sample(parent, possible_settings)
                    if new_sample:
                        new_population.append(new_sample)

                for _ in range(self.population_size - len(new_population)):
                    parent1 = random.choice(parents)
                    parent2 = random.choice(parents)
                    new_sample = self.crossover_sample(parent1, parent2)
                    new_population.append(new_sample)

                population = new_population
                t.update(1)

        return best_sample, best_valid


# Load the CSV file and extract relevant columns
df = pd.read_csv('VGG_results.csv') #  'Chicago_results_filtered.csv'  'Houston_results_filtered.csv'  'VGG_results_filtered.csv'  'LeNet_results_filtered.csv'

# Get the unique layer_settings and accuracies
possible_settings = df['layer_setting'].unique()

# Initialize the predictors
accuracy_predictor = AccuracyPredictor(df)
ece_predictor = ECEPredictor(df)
num_dropout_predictor = NumDropoutPredictor()
efficiency_predictor = EfficiencyPredictor()

# Initialize and run the EvolutionFinder
evolution_finder = EvolutionFinder(efficiency_predictor, accuracy_predictor, ece_predictor, num_dropout_predictor,
                                   population_size=20, max_time_budget=50, num_dropout_constraint=20)

best_layer_setting, best_score = evolution_finder.run_evolution_search(possible_settings, verbose=True)

print(f'Best layer_setting: {best_layer_setting}, Score (accuracy - ece - num_dropout): {best_score}')
