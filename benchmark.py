from deap import benchmarks
import NES
import PSO
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import importlib
import time

np.random.seed(42)

importlib.reload(NES) ### to make sure updated
importlib.reload(PSO)

def area_under_score(scores):
    "with a unit length scale, less is better"
    return np.sum(scores)

def compare_opt(score_NES, scores, label2="PSO"):
    "assuming similar length"
    min_iter=min(len(score_NES), len(scores))

    plt.plot(range(min_iter), score_NES[:min_iter], label="NES")
    plt.plot(range(min_iter), scores[:min_iter], label=label2)

    plt.xlabel('Iteration')
    plt.ylabel('$F(\\theta_t)$')

    plt.legend()
    plt.show()

def generate_initial(low,high):
    "Idea by us, implementation by Copilot. Sample once from on out of the 16 squares in [low,high,2]^2, do so uniformly"
    intervals = np.linspace(low, high, 5)
    # Initialize an empty list to store the samples
    samples = []
    # For each square...
    for i in range(4):
        for j in range(4):
            # Generate a random sample
            x = np.random.uniform(intervals[i], intervals[i+1])
            y = np.random.uniform(intervals[j], intervals[j+1])

            # Add the sample to the list
            samples.append((x, y))

    # Convert the list of samples to a numpy array
    samples = np.array(samples)
    return samples

def make_initia_mu_sigma():
    mus=generate_initial(-1.5,1.5)
    sigmas=np.array([[0.25, 0.25] for _ in range(16)]) # 16 squares, ~size of one square
    return mus, sigmas

def make_latex_table(dict_scores_NES, dict_scores_PSO, dict_scores_ES):
    # Initialize an empty list to store the rows of the table
    table = []

    # For each key in the dictionaries...
    for key in dict_scores_NES.keys():
        # Compute the mean and standard deviation for NES and PSO
        mean_std_NES = f"{np.mean(dict_scores_NES[key]):.2f} ± {np.std(dict_scores_NES[key]):.2f}"
        mean_std_ES = f"{np.mean(dict_scores_ES[key]):.2f} ± {np.std(dict_scores_ES[key]):.2f}"
        mean_std_PSO = f"{np.mean(dict_scores_PSO[key]):.2f} ± {np.std(dict_scores_PSO[key]):.2f}" 
        # Add a row to the table
        table.append([key, mean_std_ES, mean_std_NES, mean_std_PSO])

    # Create the LaTeX table
    latex_table = tabulate(table, headers=["Name", "ES", "NES", "PSO"], tablefmt="latex")

    # Add table, centering and label to the LaTeX table
    latex_table = "\\begin{table}[H]\n\\begin{center}\n" + latex_table + "\n\\end{center}\n\\caption{Mean and standard deviation}\n\\label{table:results}\n\\end{table}"

    print(latex_table)

def make_latex_table_speed(speed_NES, speed_PSO):
    # Compute the mean and standard deviation for NES and PSO
    mean_std_NES = f"{np.mean(speed_NES):.2f} ± {np.std(speed_NES):.2f}"
    mean_std_PSO = f"{np.mean(speed_PSO):.2f} ± {np.std(speed_PSO):.2f}"

    # Create the table data
    table = [["NES", mean_std_NES], ["PSO", mean_std_PSO]]

    # Create the LaTeX table
    latex_table = tabulate(table, headers=["Algorithm", "Speed"], tablefmt="latex")

    # Add table, centering and label to the LaTeX table
    latex_table = "\\begin{table}[H]\n\\begin{center}\n" + latex_table + "\n\\end{center}\n\\caption{The time required}\n\\label{table:speed}\n\\end{table}"

    print(latex_table)

def compare_NES_ES(mus0,sigma0,optimums):
    "detailed comparasion + tuning"

    # a comparasion of rosen and rastr is the intention #
    benchmarkS=[("sphere", benchmarks.sphere), ("rosen", benchmarks.rosenbrock), ("schaffer", benchmarks.schaffer), ("rastr", benchmarks.rastrigin)]

    bench=3
    name,neg_fitness_function=benchmarkS[bench]
    opt_problem=NES.Optimization_problem(neg_fitness_function,2,name)

    i_initial=3

    ES_scores=[]
    NES_scores=[]

    for i_initial in range(mus0.shape[0]):

        mu_init=mus0[i_initial,:]
        sigma_init=sigma0[i_initial,:]

        if bench==3:
            mus_NES, scores_NES=NES.natural_evolution_strategy(opt_problem, lr_mu=0.18, lr_sigma=0.18, initial_mu_sigma=(mu_init, sigma_init))
            mus_ES, scores_ES = NES.natural_evolution_strategy(opt_problem, lr_mu=0.0005, lr_sigma=0.0005, initial_mu_sigma=(mu_init, sigma_init), do_NES=False)
        else:
            mus_NES, scores_NES=NES.natural_evolution_strategy(opt_problem, lr_mu=0.01, lr_sigma=0.01, initial_mu_sigma=(mu_init, sigma_init))
            mus_ES, scores_ES = NES.natural_evolution_strategy(opt_problem, lr_mu=0.0002, lr_sigma=0.0002, initial_mu_sigma=(mu_init, sigma_init), do_NES=False)
        NES_scores.append(area_under_score(scores_NES))
        ES_scores.append(area_under_score(scores_ES))

        if i_initial==2:
            NES.visualize_optimization_process(mus_NES, neg_fitness_function, make_contour=True, global_opt=optimums[name], label='NES')
            NES.visualize_optimization_process(mus_ES, neg_fitness_function, make_contour=False, label='ES')
            plt.legend()
            plt.show()
            compare_opt(scores_NES, scores_ES, label2="ES")

    
    #### shows a lot of data for detailed analysis ###
    ES_scores=np.array(ES_scores)
    NES_scores=np.array(NES_scores)
    print(' ')
    print(ES_scores, NES_scores)

    #### median might be fairer
    print(np.nanmedian(ES_scores), np.nanmedian(NES_scores))
    print(' ')

    return True

def main():

    #a simple case, a badly conditioned case and 2 kinds of local minima problems
    benchmarkS=[("sphere", benchmarks.sphere), ("rosen", benchmarks.rosenbrock), ("schaffer", benchmarks.schaffer), ("rastr", benchmarks.rastrigin)]
    mus0,sigmas0=make_initia_mu_sigma()

    optimums={'rosen':[1,1], 'rastr':[0,0], 'schaffer':[0,0], 'sphere':[0,0]}

    compare_NES_ES(mus0,sigmas0, optimums)

    dict_scores_NES={name:[] for (name,_) in benchmarkS}
    dict_scores_PSO={name:[] for (name,_) in benchmarkS}
    dict_scores_ES={name:[] for (name,_) in benchmarkS}

    NES_divergences = 0
    PSO_divergences = 0
    ES_divergences = 0

    speed_PSO=[]
    speed_NES=[]

    Print=True

    for (name, neg_fitness_function) in benchmarkS:
        for (mu_init,sigma_init) in zip(mus0,sigmas0):
            opt_problem=NES.Optimization_problem(neg_fitness_function,2,name)
            # Run the NES algorithm
            if name=='rosen': #bad for NES,ES
                start=time.time()
                mus, scores_NES=NES.natural_evolution_strategy(opt_problem, lr_mu=0.01, lr_sigma=0.01, initial_mu_sigma=(mu_init, sigma_init))
                speed_NES.append(time.time()-start)
                _, scores_ES = NES.natural_evolution_strategy(opt_problem, lr_mu=0.0002, lr_sigma=0.0002, initial_mu_sigma=(mu_init, sigma_init), do_NES=False)
            elif name=='rastr': ### also badly
                start=time.time()
                mus, scores_NES=NES.natural_evolution_strategy(opt_problem, lr_mu=0.18, lr_sigma=0.18, initial_mu_sigma=(mu_init, sigma_init))
                speed_NES.append(time.time()-start)
                _, scores_ES = NES.natural_evolution_strategy(opt_problem, lr_mu=0.0005, lr_sigma=0.0005, initial_mu_sigma=(mu_init, sigma_init), do_NES=False)
            else: ### the nice cases
                start=time.time()
                mus, scores_NES=NES.natural_evolution_strategy(opt_problem, lr_mu=1, lr_sigma=1, initial_mu_sigma=(mu_init, sigma_init))
                speed_NES.append(time.time()-start)
                _, scores_ES = NES.natural_evolution_strategy(opt_problem, lr_mu=0.02, lr_sigma=0.02, initial_mu_sigma=(mu_init, sigma_init), do_NES=False)

            start=time.time()
            best_pos_PSO, scores_PSO = PSO.particle_swarm_optimization(opt_problem, initial_mus_sigma=(mu_init,sigma_init))
            speed_PSO.append(time.time()-start)

            div_ES, div_NES, div_PSO=False,False,False
            if np.isnan(area_under_score(scores_NES)):
                NES_divergences+=1
                div_NES=True
            else:
                dict_scores_NES[name].append(area_under_score(scores_NES))
            if np.isnan(area_under_score(scores_ES)):
                ES_divergences+=1
                div_ES=True
            else:
                dict_scores_ES[name].append(area_under_score(scores_ES))

            if np.isnan(area_under_score(scores_PSO)):
                PSO_divergences+=1
                div_PSO=True
            else:
                dict_scores_PSO[name].append(area_under_score(scores_PSO))


            if Print and len(dict_scores_NES[name])==1 and not div_NES and not div_PSO: #detailed analysis enabled here
                NES.visualize_optimization_process(mus, neg_fitness_function, make_contour=True, global_opt=optimums[name], label='NES')
                NES.visualize_optimization_process(best_pos_PSO, neg_fitness_function, make_contour=False, label='PSO')
                plt.legend()
                plt.show()
                compare_opt(scores_NES, scores_PSO)
    
    print(" ")

    print("PSO: ", PSO_divergences)
    print("NES: ", NES_divergences)
    print("ES: ", ES_divergences)

    print("")

    make_latex_table(dict_scores_NES, dict_scores_PSO, dict_scores_ES)

    print(" ")

    speed_PSO=np.array(speed_PSO)
    speed_NES=np.array(speed_NES)

    make_latex_table_speed(speed_NES,speed_PSO) 

main()