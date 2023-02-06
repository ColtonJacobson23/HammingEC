#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
First attempt at using EC for creating "best alignments" between two protein sequences
Should also apply to other sequence types
Notes regarding this EC and it's configuration
1. Representation
 - Sequences for any biological information encoding will be done with strings
 - One sequence will be a Target sequence while the rest will be variations of another sequence
 - One thing to consider is maybe representing the sequence (atleast for proteins) not as the total
   protein sequence but rather just where spaces go and how long they are, then you only have to
   deal with spaces as maybe a dictionary for location and number. 
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

# %%
import random
from typing import TypedDict
from typing import Optional
import copy
import math
from collections import Counter
import string


nums = []
for x in range(-100, 101):
    nums.append(str(x))


class Individual(TypedDict):
    """Type of each individual to evolve"""

    genome: string
    fitness: float


Population = list[Individual]


def gen_rand_seq(len: int) -> str:
    return ""


'''
With creating a new individual, the issue becomes, do you
take the given genome and add random spaces now, or do you 
let mutation and recombination do that after?
'''
def initialize_individual(genome: str, fitness: float) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as Node, fitness as integer (lower = better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[Node, int]
    Modifies:       Nothing
    Calls:          Basic python only
    """
    return {"genome": genome, "fitness": fitness}


def initialize_pop(pop_size: int, io_data: IOdata) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Goal string, population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    Calls:          random.choice-1, string.ascii_letters-1, initialize_individual-n
    Example doctest:
    """
    population: Population = []
    for x in range (pop_size):
        individual = initialize_individual(
            genome=gen_rand_seq(), 
            fitness=0
            )
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
    Calls:          Basic python, random.choice-1, initialize_individual-2
    Example doctest:
    """
    nlist1 = []
    nlist2 = []
    rand_node1 = []
    rand_node2 = []

    make_stack(parent1["genome"], nlist1)
    make_stack(parent2["genome"], nlist2)
    select_rand_node(parent1["genome"], rand_node1)
    select_rand_node(parent2["genome"], rand_node2, tree_depth(rand_node1[0][0]))
    new_subtree1 = []
    make_stack(rand_node1[0][0], new_subtree1)
    new_subtree2 = []
    make_stack(rand_node2[0][0], new_subtree2)
    rand_index1 = rand_node1[0][1]-1
    rand_index2 = rand_node2[0][1]-1


    orig1 = nlist1[:rand_index1]
    change1 = nlist1[rand_index1:]
    depth1 = 1
    count1 = 0
    x = change1[count1]
    while(x not in nums):
        x = change1[count1]
        depth1 += 1
        count1 += 1
    finished_tree1 = orig1 + new_subtree2
    p1 = change1[len(new_subtree1):]
    finished_tree1 = finished_tree1 + p1
    ch1 = " ".join(finished_tree1)

    orig2 = nlist2[:rand_index2]
    change2 = nlist2[rand_index2:]
    depth2 = 1
    count2 = 0
    x = change2[count2]
    while(x not in nums):
        x = change2[count2]
        depth2 += 1
        count2 += 1
    finished_tree2 = orig2 + new_subtree1
    p2 = change2[len(new_subtree2):]
    finished_tree2 = finished_tree2 + p2
    ch2 = " ".join(finished_tree2)



    if ch1.count("x") == 0:
        put_an_x_in_it(ch1)
    elif ch1.count("x") == 2:
        ind = ch1.find("x")
        ch1 = list(ch1)
        ch1[ind] = str(random.randint(-100, 100))
        ch1 = "".join(ch1)

    if ch2.count("x") == 0:
        put_an_x_in_it(ch2)
    elif ch2.count("x") == 2:
        ind = ch2.find("x")
        ch2 = list(ch2)
        ch2[ind] = str(random.randint(-100, 100))
        ch2 = "".join(ch2)
    
    child1 = initialize_individual(ch1, 0)
    child2 = initialize_individual(ch2, 0)

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
    Calls:          Basic python, random.random~n/2, recombine pair-n
    """
    combination: Population = []
    for ipair in range(0, len(parents) - 1, 2):
        if random.random() < recombine_rate:
            try:
                child1, child2 = recombine_pair(
                    parent1=parents[ipair], parent2=parents[ipair + 1]
                )
            except:
                child1, child2 = parents[ipair], parents[ipair + 1]
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
    Calls:          Basic python, random,choice-1,
    Example doctest:
    """
    if random.random() < mutate_rate:
        nlist = []
        make_stack(parent["genome"], nlist)
        rand_index = random.randint(0, len(nlist) - 1)
        orig = nlist[:rand_index]
        change = nlist[rand_index:]
        depth = 1
        count = 0
        x = change[count]
        while(x not in nums):
            depth += 1
            count += 1
            x = change[count]
        new_subtree = gen_rand_prefix_code(depth).split(" ")
        finished_tree = orig + new_subtree
        p2 = change[len(new_subtree):]
        finished_tree = finished_tree + p2
        mutant = initialize_individual(" ".join(finished_tree), 0)
        return mutant
    return parent
        

def mutate_group(children: Population, mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, mutate_individual-n
    Example doctest:
    """
    new_group: Population = []
    for child in children:
        try:
            new_group.append(mutate_individual(parent=child, mutate_rate=mutate_rate))
        except:
            new_group.append(child)
    return new_group


# DO NOT MODIFY >>>>
def evaluate_individual(individual: Individual, io_data: IOdata) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
    Parameters:     One Individual, data formatted as IOdata
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Notes:          train/test format is like PSB2 (see IOdata above)
    Example doctest:
    >>> evaluate_individual(ind1, io_data)
    """
    fitness = 0
    errors = []
    for sub_eval in io_data:
        eval_string = parse_tree_return(individual["genome"]).replace(
            "x", str(sub_eval["input1"])
        )

        # In clojure, this is really slow with subprocess
        # eval_string = "( float " + eval_string + ")"
        # returnobject = subprocess.run(
        #     ["clojure", "-e", eval_string], capture_output=True
        # )
        # result = float(returnobject.stdout.decode().strip())

        # In python, this is MUCH MUCH faster:
        try:
            y = eval(prefix_to_infix(eval_string))
        except ZeroDivisionError:
            y = math.inf

        errors.append(abs(sub_eval["output1"] - y))
    # Higher errors is bad, and longer strings is bad
    fitness = sum(errors) + len(eval_string.split())
    # Higher fitness is worse
    individual["fitness"] = fitness


# <<<< DO NOT MODIFY


def evaluate_group(individuals: Population, io_data: IOdata) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          Basic python, evaluate_individual-n
    Example doctest:
    """
    for i in range(len(individuals)):
        evaluate_individual(individual=individuals[i], io_data=io_data)


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          Basic python only
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
    Calls:          Basic python, random.choices-1
    Example doctest:
    """
    parents: Population = []
    indiv = []
    while individuals:
        x = individuals.pop()
        if x["fitness"] != math.inf:
            indiv.append(x)
    fitnesses = [i["fitness"] for i in indiv]
    parents = random.choices(indiv, fitnesses, k=number)
    return parents


def survivor_select(individuals: Population, pop_size: int, io_data: IOdata) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          Basic python only
    Example doctest:
    """
    new_pop = initialize_pop(int(pop_size/2), io_data)
    evaluate_group(new_pop, io_data)
    return individuals[:int(pop_size/2)] + new_pop
    # return individuals[:pop_size]


def evolve(io_data: IOdata, pop_size: int = 1000) -> Population:
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
    # pop_size: int = 100
    print("Starting to evolve-----------------------------------------------------------------------")
    population = initialize_pop(pop_size, io_data) 
    evaluate_group(population, io_data) 
    rank_group(population) 
    best_fitness = population[0]['fitness'] 
    perfect_fitness = 1
    counter = 0
    while best_fitness > 300:
        counter += 1 
        parents = parent_select(individuals=population, number=pop_size) 
        # for x in parents:
        #     if tree_depth(x["genome"]) != 4:
        #         x = initialize_individual(genome=gen_rand_prefix_code(4), fitness=0)
        children = recombine_group(parents=parents, recombine_rate=0.7) 
        # mutate_rate = (1 - best_fitness / perfect_fitness) / 5 
        mutants = mutate_group(children=children, mutate_rate=0.7) 
        evaluate_group(individuals=mutants, io_data=io_data) 
        everyone = population + mutants 
        rank_group(individuals=everyone) 
        population = survivor_select(individuals=everyone, pop_size=pop_size, io_data=io_data)
        #if best_fitness != population[0]['fitness']:
        best_fitness = population[0]['fitness'] 
        print('Iteration number', counter, 'with best individual', population[0], "of depth", tree_depth(population[0]["genome"]))


    return population

if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)
    random.seed(42)

    print(divider)
    print("Number of possible genetic programs: infinite...")
    print("Lower fitness is better.")
    print(divider)

    X = list(range(-10, 110, 10))
    Y = [(x * (9 / 5)) + 32 for x in X]
    # data = [{"input1": x, "output1": y} for x, y in zip(X, Y)]
    # mypy wanted this:
    data = [initialize_data(input1=x, output1=y) for x, y in zip(X, Y)]

    # Correct:
    print("Example of celcius to farenheight:")
    ind1 = initialize_individual("( + ( * x ( / 9 5 ) ) 32 )", 0)
    evaluate_individual(ind1, data)
    print_tree(ind1["genome"])
    print("Fitness", ind1["fitness"])

    # Yours
    train = data[: int(len(data) / 2)]
    test = data[int(len(data) / 2) :]
    population = evolve(train)
    evaluate_individual(population[0], test)
    population[0]["fitness"]

    print("Here is the best program:")
    parse_tree_print(population[0]["genome"])
    print("And it's fitness:")
    print(population[0]["fitness"])