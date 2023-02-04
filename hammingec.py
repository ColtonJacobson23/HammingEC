'''
First attempt at using EC for creating "best alignments" between two protein sequences
Should also apply to other sequence types
Notes regarding this EC and it's configuration
1. Representation
 - Sequences for any biological information encoding will be done with strings
 - One sequence will be a Target sequence while the rest will be variations of another sequence
2. Fitness function
- It will use the hamming distance calculation (for now, possible implementations of dot matrix are in the works)
- Fitness will be the score of the hamming calculation against the Target sequence
3. Target vs non-target
- The goal of this program is to use EC to build a protein string that gets the score
    closest to the target hamming score/percent 
    For example, you might only want an 80-90% similarity as opposed to the highest possible score
    For more info see 4. Optimization
- The user will supply a target protein sequence and the one we are trying to align to it (the non-target sequence)
- The program will then perform the functions of EC on the non-target sequence until it has satisfied our target hamming distance
4. Optimization
- One key feature of optimization will be a target hamming distance
- The user can set the target hamming distance to whatever percent difference they want the program to meet
'''
#!/usr/bin/python3
# -*- coding: utf-8 -*-

# %%
import random
from typing import TypedDict
import math
import json
import string

# DO NOT MODIFY >>>>
# First, what should our representation look like?
# Is there any modularity in adjacency?
# What mechanisms capitalize on such modular patterns?
# ./corpus/2_count.py specificies this same structure
# Positions    01234   56789   01234
LEFT_DVORAK = "',.PY" "AOEUI" ";QJKX"
LEFT_QWERTY = "QWERT" "ASDFG" "ZXCVB"
LEFT_COLEMK = "QWFPG" "ARSTD" "ZXCVB"
LEFT_WORKMN = "QDRWB" "ASHTG" "ZXMCV"

LEFT_DISTAN = "11111" "00001" "11111"
LEFT_ERGONO = "00001" "00001" "11212"
LEFT_EDGE_B = "01234" "01234" "01234"

# Positions     56   7890123   456789   01234
RIGHT_DVORAK = "[]" "FGCRL/=" "DHTNS-" "BMWVZ"
RIGHT_QWERTY = "-=" "YUIOP[]" "HJKL;'" "NM,./"
RIGHT_COLEMK = "-=" "JLUY;[]" "HNEIO'" "KM,./"
RIGHT_WOKRMN = "-=" "JFUP;[]" "YNEOI'" "KL,./"

RIGHT_DISTAN = "23" "1111112" "100000" "11111"
RIGHT_ERGONO = "22" "2000023" "100001" "10111"
RIGHT_EDGE_B = "10" "6543210" "543210" "43210"

DVORAK = LEFT_DVORAK + RIGHT_DVORAK
QWERTY = LEFT_QWERTY + RIGHT_QWERTY
COLEMAK = LEFT_COLEMK + RIGHT_COLEMK
WORKMAN = LEFT_WORKMN + RIGHT_WOKRMN

DISTANCE = LEFT_DISTAN + RIGHT_DISTAN
ERGONOMICS = LEFT_ERGONO + RIGHT_ERGONO
PREFER_EDGES = LEFT_EDGE_B + RIGHT_EDGE_B

# Real data on w.p.m. for each letter, normalized.
# Higher values is better (higher w.p.m.)
with open(file="typing_data/manual-typing-data_qwerty.json", mode="r") as f:
    data_qwerty = json.load(fp=f)
with open(file="typing_data/manual-typing-data_dvorak.json", mode="r") as f:
    data_dvorak = json.load(fp=f)
data_values = list(data_qwerty.values()) + list(data_dvorak.values())
mean_value = sum(data_values) / len(data_values)
data_combine = []
for dv, qw in zip(DVORAK, QWERTY):
    if dv in data_dvorak.keys() and qw in data_qwerty.keys():
        data_combine.append((data_dvorak[dv] + data_qwerty[qw]) / 2)
    if dv in data_dvorak.keys() and qw not in data_qwerty.keys():
        data_combine.append(data_dvorak[dv])
    if dv not in data_dvorak.keys() and qw in data_qwerty.keys():
        data_combine.append(data_qwerty[qw])
    else:
        # Fill missing data with the mean
        data_combine.append(mean_value)


class Individual(TypedDict):
    genome: str
    fitness: int


Population = list[Individual]


def print_keyboard(individual: Individual) -> None:
    layout = individual["genome"]
    fitness = individual["fitness"]
    """Prints the keyboard in a nice way"""
    print("______________  ________________")
    print(" ` 1 2 3 4 5 6  7 8 9 0 " + " ".join(layout[15:17]) + " Back")
    print("Tab " + " ".join(layout[0:5]) + "  " + " ".join(layout[17:24]) + " \\")
    print("Caps " + " ".join(layout[5:10]) + "  " + " ".join(layout[24:30]) + " Enter")
    print(
        "Shift " + " ".join(layout[10:15]) + "  " + " ".join(layout[30:35]) + " Shift"
    )
    print(f"\nAbove keyboard has fitness of: {fitness}")


# <<<< DO NOT MODIFY


def initialize_individual(genome: str, fitness: int) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as string, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    return {"genome": genome, "fitness": fitness}


def initialize_pop(example_genome: str, pop_size: int) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Goal string, population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    myChars = list(DVORAK)
    testChars = myChars
    population: Population = []
    for x in range (pop_size):
        random.shuffle(testChars)
        new_gen = "".join(testChars)
        individual = initialize_individual(genome=new_gen, fitness=0)
        population.append(individual)
    return population


def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    index = random.choice(range(len(parent1["genome"])))
    #gen1 = list(parent1["genome"])
    #gen2 = list(parent2["genome"])
    
    if index > len(parent1["genome"]) / 2:
        new_gen1 = parent1["genome"][:index]
        new_gen2 = parent2["genome"][:index]
    else:
        new_gen1 = parent1["genome"][index:]
        new_gen2 = parent2["genome"][index:]

    for x in range(len(parent1["genome"])):
        if parent1["genome"][x] not in new_gen2:
            new_gen2 += parent1["genome"][x]
        if parent2["genome"][x] not in new_gen1:
            new_gen1 += parent2["genome"][x]
        

    #new_genome1 = "".join(gen3)
    #new_genome2 = "".join(gen4)
    child1 = initialize_individual(genome=new_gen1, fitness=0)
    child2 = initialize_individual(genome=new_gen2, fitness=0)
    return [child1, child2]


def recombine_group(parents: Population, recombine_rate: float) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 2-3, 3-4, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     parents and recombine rate
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    Calls:          ?
    """
    combination: Population = []
    for ipair in range(0, len(parents) - 1, 2):
        if random.random() < recombine_rate:
            child1, child2 = recombine_pair(
                parent1=parents[ipair], parent2=parents[ipair + 1]
            )
        else:
            child1, child2 = parents[ipair], parents[ipair + 1]
        combination.extend([child1, child2])
    return combination 

def mutate_individual(parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    new_genome = parent["genome"].split()
    for ch in new_genome:
        if random.random() < mutate_rate:
            temp = random.choice(new_genome)
            tempInd = new_genome.index(ch)
            tempInd2 = new_genome.index(temp)
            new_genome[tempInd] = temp
            new_genome[tempInd2] = ch

    gen_str = ""
    for ch in parent["genome"]:
        gen_str = gen_str + ch
    mutant = initialize_individual(genome=gen_str, fitness=0)
    return mutant


def mutate_group(children: Population, mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    new_group: Population = []
    for child in children:
        new_group.append(mutate_individual(parent=child, mutate_rate=mutate_rate))
    return new_group


# DO NOT MODIFY >>>>
def evaluate_individual(individual: Individual) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
                    Assumes and relies on the logc of ./corpus/2_counts.py
    Parameters:     One Individual
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Example doctest:
    """
    layout = individual["genome"]

    # Basic return to home row, with no extra cost for repeats.
    fitness = 0
    for key in layout:
        fitness += count_dict[key] * int(DISTANCE[layout.find(key)])

    # Vowels on the left, Consosants on the right
    for pos, key in enumerate(layout):
        if key in "AEIOUY" and pos > 14:
            fitness += 1

    # Top-down guess at ideal ergonomics
    for key in layout:
        fitness += count_dict[key] * int(ERGONOMICS[layout.find(key)])

    # [] {} () <> should be adjacent.
    # () ar fixed by design choice (number line).
    # [] and {} are on same keys.
    # Perhaps ideally, <> and () should be on same keys too...
    right_edges = [4, 9, 14, 16, 23, 29, 34]
    for pos, key in enumerate(layout):
        # order of (x or y) protects index on far right:
        if key == "[" and (pos in right_edges or "]" != layout[pos + 1]):
            fitness += 1
        if key == "," and (pos in right_edges or "." != layout[pos + 1]):
            fitness += 1

    # Symbols should be toward edges.
    for pos, key in enumerate(layout):
        if key in "-[],.';/=":
            fitness += int(PREFER_EDGES[pos])

    # Keybr.com querty-dvorak average data as estimate of real hand
    for pos, key in enumerate(layout):
        fitness += count_dict[key] / data_combine[pos]

    # Shortcut characters (skip this one).
    # On right hand for keyboarders (left ctrl is usually used)
    # On left hand for mousers (for one-handed shortcuts).
    pass

    individual["fitness"] = fitness


# <<<< DO NOT MODIFY


def evaluate_group(individuals: Population) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          ?
    Example doctest:
    """
    for i in range(len(individuals)):
        evaluate_individual(individual=individuals[i])


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          ?
    Example doctest:
    """
    individuals.sort(key=lambda ind: ind["fitness"], reverse=False)


def parent_select(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    parents: Population = []
    fitnesses = [i["fitness"] for i in individuals]
    parents = random.choices(individuals, fitnesses, k=number)
    return parents


def survivor_select(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    new_pop = initialize_pop(DVORAK, pop_size)
    evaluate_group(new_pop)
    return individuals[:int(pop_size/2)] + new_pop
    #return individuals[:pop_size]
    


def evolve(example_genome: str) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    Calls:          Basic python, all your functions
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller
    pop_size: int = 15000
    population = initialize_pop(example_genome, pop_size) 
    evaluate_group(population) 
    rank_group(population) 
    best_fitness = population[0]['fitness'] 
    perfect_fitness = len(example_genome) 
    counter = 0
    while best_fitness > 20:
        counter += 1 
        parents = parent_select(individuals=population, number=pop_size) 
        children = recombine_group(parents=parents, recombine_rate=0.9) 
        mutate_rate = (1 - best_fitness / perfect_fitness) / 5 
        mutants = mutate_group(children=children, mutate_rate=0.7) 
        evaluate_group(individuals=mutants) 
        everyone = population + mutants 
        rank_group(individuals=everyone) 
        population = survivor_select(individuals=everyone, pop_size=pop_size)
        #if best_fitness != population[0]['fitness']:
        best_fitness = population[0]['fitness'] 
        print('Iteration number', counter, 'with best individual', population[0])


    return population


# Seed for base grade.
# For the exploratory competition points (last 10),
# comment this one line out if you want, but put it back please.
seed = True

# DO NOT MODIFY >>>>
if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)

    if seed:
        random.seed(42)

    with open("corpus/counts.json") as fhand:
        count_dict = json.load(fhand)

    # print("Counts of characters in big corpus, ordered by freqency:")
    # ordered = sorted(count_dict, key=count_dict.__getitem__, reverse=True)
    # for key in ordered:
    #     print(key, count_dict[key])

    print(divider)
    print(
        f"Number of possible permutations of standard keyboard: {math.factorial(len(DVORAK)):,e}"
    )
    print("That's a huge space to search through")
    print("The messy landscape is a difficult to optimize multi-modal space")
    print("Lower fitness is better.")

    print(divider)
    print("\nThis is the Dvorak keyboard:")
    dvorak = Individual(genome=DVORAK, fitness=0)
    evaluate_individual(dvorak)
    print_keyboard(dvorak)

    print(divider)
    print("\nThis is the Workman keyboard:")
    workman = Individual(genome=WORKMAN, fitness=0)
    evaluate_individual(workman)
    print_keyboard(workman)

    print(divider)
    print("\nThis is the Colemak keyboard:")
    colemak = Individual(genome=COLEMAK, fitness=0)
    evaluate_individual(colemak)
    print_keyboard(colemak)

    print(divider)
    print("\nThis is the QWERTY keyboard:")
    qwerty = Individual(genome=QWERTY, fitness=0)
    evaluate_individual(qwerty)
    print_keyboard(qwerty)

    print(divider)
    print("\nThis is a random layout:")
    badarr = list(DVORAK)
    random.shuffle(badarr)
    badstr = "".join(badarr)
    badkey = Individual(genome=badstr, fitness=0)
    evaluate_individual(badkey)
    print_keyboard(badkey)

    print(divider)
    input("Press any key to start")
    population = evolve(example_genome=DVORAK)

    print("Here is the best layout:")
    print_keyboard(population[0])

    grade = 0
    if qwerty["fitness"] < population[0]["fitness"]:
        grade = 0
    if colemak["fitness"] < population[0]["fitness"]:
        grade = 50
    if workman["fitness"] < population[0]["fitness"]:
        grade = 60
    elif dvorak["fitness"] < population[0]["fitness"]:
        grade = 70
    else:
        grade = 80

    with open(file="results.txt", mode="w") as f:
        f.write(str(grade))

    with open(file="best_ever.txt", mode="r") as f:
        past_record = f.readlines()[1]
    if population[0]["fitness"] < float(past_record):
        with open(file="best_ever.txt", mode="w") as f:
            f.write(population[0]["genome"] + "\n")
            f.write(str(population[0]["fitness"]))
# <<<< DO NOT MODIFY
