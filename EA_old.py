import math
import sys
import numpy as np
import random

# Parameters
global sel_type
global recomb_type
global mut_type

global parents_size
global offspring_size

# Fallback parameter
global fallback_patience


global verbose
global curr_seed 
global default_budget


# Parameters
params = { 
            'func': None,
            'sel_type': 'm+l',
            'recomb_type': 'intermediate',
            'mut_type': 'custom_sigma',
            'parents_size': 6,
            'offspring_size': 80,
            'budget': 50000,
            'fallback_patience': 1000,
            'verbose': 0
        }
verbose = 0
write_results=0
curr_seed = 0
np.random.seed(curr_seed)



def Recombination(  parents, 
                    sigmas=[], 
                    angles=[], 
                    recomb_type='intermediate', 
                    offspring_size=80
                ):
    """
    Recombine individuals.

    recomb_type: 'discrete' or 'intermediate' or 'global_intermediary' or 'global_intermediate'
    """
    
    # parents length check
    len_parents = len(parents)
    if len_parents < 2:
        return parents

    if recomb_type == 'global_intermediary':

        parents_temp = parents.copy()
        curr_n_offspring = 0
        offspring = []
        new_sigs = []
        new_angles = []
        while curr_n_offspring < offspring_size:
            new_child = np.array(np.sum(parents_temp, axis=0) / len(parents_temp))
            offspring.append(new_child)
            # We do this when we want more than 1 offspring for compatibility
            parents_temp = np.concatenate((parents_temp,[new_child]),axis=0)

            # Calculate offspring sigmas
            new_sigma = np.sum(sigmas, axis=0) / len(sigmas)
            new_sigs.append(new_sigma)

            # Calculate offspring angles
            new_angle = np.sum(angles, axis=0) / len(angles)
            new_angles.append(new_angle)

            curr_n_offspring += 1
        
        return np.array(offspring), np.array(new_sigs), np.array(new_angles)

    elif recomb_type == 'global_intermediate':

        individual_len = parents.shape[1]
        idxes = np.arange(0,len_parents)
        offspring = []
        off_sigs = []

        curr_n_offspring = 0
        while curr_n_offspring < offspring_size:
            off_temp = []
            sig_temp = []
            for i in range(individual_len):
                parents_idx = np.random.choice(idxes,2,replace=False)
                parent_1 = parents[parents_idx[0]]
                parent_2 = parents[parents_idx[1]]

                curr_gene = (parent_1[i] + parent_2[i])/2
                off_temp.append(curr_gene)

                curr_sig = (sigmas[parents_idx[0]][i] + sigmas[parents_idx[1]][i])/2
                sig_temp.append(curr_sig)
            
            offspring.append(off_temp)
            off_sigs.append(sig_temp)
            curr_n_offspring += 1
        
        # Initialize new angles for offspring
        off_angles = np.deg2rad(np.random.uniform(0,360, (offspring_size,angles.shape[1])))
        return np.array(offspring), np.array(off_sigs), off_angles


    elif recomb_type == 'intermediate':

        idxes = np.arange(0,len_parents)
        curr_n_offspring = 0
        offspring = []
        off_sigs = []
        off_angles = []
        while curr_n_offspring < offspring_size:
            # Choose random parents
            parents_idx = np.random.choice(idxes,2,replace=False)
            parent_1 = parents[parents_idx[0]]
            parent_2 = parents[parents_idx[1]]
            # Append child to offspring
            child = [(parent_1[i]+parent_2[i])/2 for i in range(len(parent_1))]
            offspring.append(child)

            # Create new sigmas for offspring
            curr_sigs = [sigmas[parents_idx[0]], sigmas[parents_idx[1]]]
            child_sig = np.average(curr_sigs, axis=0)
            off_sigs.append(child_sig)

            # Create new angles for offspring
            curr_angles = [angles[parents_idx[0]], angles[parents_idx[1]]]
            child_ang = np.average(curr_angles, axis=0)
            off_angles.append(child_ang)

            curr_n_offspring += 1
        
        return np.array(offspring), np.array(off_sigs), np.array(off_angles)


    elif recomb_type == 'discrete':
        p=0.5
        # Simply permute the parents to simulate randomness
        parents_permut = np.random.permutation(parents)
        sigmas_permut = sigmas.copy()
        

        # Iterate over pair of parent permutations
        parent_idx=0
        while parent_idx < len_parents:
            parent_1 = parents_permut[parent_idx]
            parent_2 = parents_permut[parent_idx+1]

            recomb_points = np.arange(1,len(parent_1)-1)
            n_recombs = len(recomb_points)

            # Check which recombinations to apply
            apply_crossover = np.array([True, False])
            probs = [p, 1-p]
            choices = np.random.choice(a=apply_crossover, p=probs, size=n_recombs)
            recomb_points = recomb_points[recomb_points*choices != 0]

            # Add first and last index for convenience to cycle through later
            recomb_points = np.insert(recomb_points,0,0)
            recomb_points = np.append(recomb_points,[len(parent_1)-1])
            len_recombs=len(recomb_points)

            recomb_idx=0
            # Cycle through recombination_points
            while recomb_idx < len_recombs-1:
                point_start = recomb_points[recomb_idx]
                point_end = recomb_points[recomb_idx+1]

                # Inplace change values 
                if  recomb_idx % 2 == 0:
                    temp = parent_1[point_start:point_end].copy()
                    parent_1[point_start:point_end]  = parent_2[point_start:point_end]
                    parent_2[point_start:point_end] = temp

                recomb_idx +=1
            parent_idx += 2
        return parents_permut, sigmas_permut, angles
    else:
        print("No recombination selected!!")



def Mutation(   offspring, 
                sigmas=[],
                angles=[], 
                mut_type='custom_sigma'
            ):
    """
    Mutate the individual.

    mut_type: 'custom_sigma' or 'individual_sigma' or 'correlated'
    """

    range_individuals = range(offspring.shape[0])
    len_individual = offspring.shape[1]

    # Learning rates constant
    lr = 1/np.sqrt(2*(np.sqrt(len_individual)))
    lr_prime = 1/(np.sqrt(2*len_individual))

    if mut_type == 'custom_sigma':
        # Iterate over individuals
        for individual in range_individuals:
            curr_ind = offspring[individual]
            curr_sigma = sigmas[individual]

            # Iterate over sigmas
            for sigma_i in range(len_individual):
                # Update current sigma
                normal_matr_prime = np.random.normal(0,lr_prime,1)
                normal_matr = np.random.normal(0,lr,1)
                curr_sigma[sigma_i] = curr_sigma[sigma_i]*(
                        np.exp(normal_matr+normal_matr_prime))

                # Mutate individual
                sigma_noise = np.random.normal(0,curr_sigma[sigma_i],1)
                curr_ind[sigma_i] = curr_ind[sigma_i] + sigma_noise

        return offspring, sigmas, angles


    elif mut_type == 'individual_sigma':
        # Iterate over individuals
        for individual in range_individuals:
            curr_ind = offspring[individual]
            curr_sigma = sigmas[individual]

            normal_matr_prime = np.random.normal(0,lr_prime,1)
            # Iterate over sigmas
            for sigma_i in range(len_individual):
                # Update current sigma
                normal_matr = np.random.normal(0,lr,1)
                curr_sigma[sigma_i] = curr_sigma[sigma_i]*(
                        np.exp(normal_matr+normal_matr_prime))

                # Mutate individual
                sigma_noise = np.random.normal(0,curr_sigma[sigma_i],1)
                curr_ind[sigma_i] = curr_ind[sigma_i] + sigma_noise

        return offspring, sigmas, angles


    elif mut_type == 'correlated':
        # Starting params
        angles_len = angles.shape[1]
        beta = math.pi/36

        for individual in range_individuals:

            curr_sigma = sigmas[individual]
            normal_matr_prime = np.random.normal(0,lr_prime,1)
            # Update our sigma
            for sigma_i in range(len_individual):
                normal_matr = np.random.normal(0,lr,1)
                curr_sigma[sigma_i] = curr_sigma[sigma_i]*(
                        np.exp(normal_matr+normal_matr_prime))

            # Update angles
            angles_noise = np.random.normal(0,beta,angles_len)
            angles = angles + angles_noise

            angles[angles > math.pi] = angles[angles > math.pi] - 2*math.pi*np.sign(angles[angles > math.pi])

            # Calculate C matrix
            count = 0
            C = np.identity(len_individual)
            for i in range(len_individual-1):
                for j in range(i+1,len_individual):
                    R = np.identity(len_individual)
                    R[i,i] = math.cos(angles[individual][count])
                    R[j,j] = math.cos(angles[individual][count])
                    R[i,j] = -math.sin(angles[individual][count])
                    R[j,i] = math.sin(angles[individual][count])
                    C = np.dot(C, R)
                    count += 1
            s = np.identity(len_individual)
            np.fill_diagonal(s, sigmas[individual])
            C = np.dot(C, s)
            C = np.dot(C, C.T)
            
            # Update offspring
            sigma_std = np.random.multivariate_normal(mean=np.full((len_individual),fill_value=0), cov=C)
            fix = np.array([ random.gauss(0,i) for i in sigma_std ])
            offspring[individual] =  offspring[individual] + fix
        return offspring, sigmas, angles

    else:
        print("No mutation selected!")



def Selection(  population,
                evaluations, 
                sigmas=[],
                angles=[],
                sel_type='1+l',
                parents_size=6
            ):
    """
    returns a new population created by consecutive selections 
    of the parent population with repetition.

    sel_type: can be '1+l' or '1,l'
    """
    # Find the best out of the offspring
    if sel_type == 'm,l':
        # Consider only offspring of population
        evals_only_offspring = evaluations[parents_size:]
        only_offspring_pop = population[parents_size:]
        only_offspring_sigmas = sigmas[parents_size:]
        only_offspring_angles = angles[parents_size:]

        # Choose best offspring
        indexes = evals_only_offspring.argsort()[:parents_size]

        # Get new variables
        new_pop = np.array([only_offspring_pop[index] for index in indexes])
        new_evals = np.array([evals_only_offspring[index] for index in indexes])
        new_sigmas = np.array([only_offspring_sigmas[index] for index in indexes])
        new_angles = np.array([only_offspring_angles[index] for index in indexes])
        return new_pop, new_evals, new_sigmas, new_angles

    
    # Find the best out of the parents and offspring
    elif sel_type == 'm+l':
        #get best indexes
        indexes = evaluations.argsort()[:parents_size]

        new_pop = np.array([population[index] for index in indexes])
        new_evals = np.array([evaluations[index] for index in indexes])
        new_sigmas = np.array([sigmas[index] for index in indexes])
        new_angles = np.array([angles[index] for index in indexes])
        
        return new_pop, new_evals, new_sigmas, new_angles

    else:
        print("No Selection selected!")



def ES(func, budget = 50000,
        parents_size = 6,offspring_size = 80,
        recomb_type='global_intermediate',
        mut_type='custom_sigma',
        sel_type='1+l',
        fallback_patience = 1000,
        verbose=1):
    """
    Main Evolutionary Strategy algorithm
    """

    # Temporary variables
    n_variables = func.meta_data.n_variables
    best_eval = sys.float_info.max
    best_individual = []
    curr_budget = 0
    # Fallback variables
    curr_patience = 0
    fallback_mode = False

    # Generate parents
    population = np.random.uniform(0,1,size=(parents_size, n_variables))

    # Initialize Sigmas and angles for individual sigma mutation
    # and correlated mutation strategies
    sigmas = np.array(np.random.uniform(0.,1,population.shape))
    angles_len = int((n_variables*(n_variables-1))/2)
    angles = np.deg2rad(np.random.uniform(0,360, (parents_size,angles_len)))

    # Initial evaluations
    evals = np.array([func(i) for i in population[-parents_size :]])
    curr_budget += parents_size
    curr_patience += parents_size
    
    while curr_budget < budget:

        # Keep track of best fit individual and evaluation
        min_eval = evals.min()
        if best_eval > min_eval:
            best_eval = min_eval
            best_individual = population[evals.argmin()].copy()
            budget_for_max = curr_budget
            curr_patience = 0
            if verbose > 1:
                print(f"new best val: {best_eval}, used budget: {budget_for_max}")
    
        #Recombine and Mutate offspring
        offspring, off_sigmas, off_angles = Recombination(parents=population, sigmas=sigmas, 
                                                        angles=angles, recomb_type=recomb_type,
                                                        offspring_size=offspring_size)
        offspring, off_sigmas, off_angles = Mutation(offspring, mut_type=mut_type,
                                            sigmas=off_sigmas, angles=off_angles)
        
        
        # Evaluate offspring, stop when reaching 50000 budget
        offspring_fitness =[]
        for off in offspring:
            if curr_budget >= 50000:
                if verbose > 0:
                    print(f"--Target: {func.objective.y}, \t found: {round(best_eval,4)}, \t budget required: {budget_for_max},")
                return best_individual, best_eval
            else:
                curr_budget = curr_budget + 1
                curr_patience += 1
                offspring_fitness.append(func(off))
        offspring_fitness = np.array(offspring_fitness)

        # Concatenate new offspring variables to parents
        evals = np.concatenate((evals,offspring_fitness),axis=0)
        population = np.concatenate((population,offspring),axis=0)
        sigmas = np.concatenate((sigmas,off_sigmas),axis=0)
        angles = np.concatenate((angles,off_angles), axis=0)

        # Selection step with fallback
        if fallback_mode:
            population, evals, sigmas, angles = Selection(population,evals,sigmas,angles,sel_type='m,l', parents_size=parents_size)
        else:
            population, evals, sigmas, angles = Selection(population,evals,sigmas,angles,sel_type=sel_type, parents_size=parents_size)

        # Switch to fallback mode and back to normal
        if ((curr_patience > fallback_patience) or (fallback_mode == True) ) :
            if fallback_mode == True:
                fallback_mode = False
                curr_patience = 0
                if verbose > 1:
                    print(f"+++ Deactivated fallback! Budget: {curr_budget} +++")
            else:
                fallback_mode = True
                curr_patience = 0
                if verbose > 1:
                    print(f"+++ Activated fallback! Budget: {curr_budget} +++")

    # print results
    if verbose > 0:
        print(f"--Target: {func.objective.y}, \t found: {round(best_eval,4)}, \t budget required: {budget_for_max},")

    return best_individual, best_eval
