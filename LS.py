import numpy as np
import glob
import numpy as np
from scipy.sparse import load_npz
import random
from  numba import njit
import os
import pandas as pd
from multiprocessing.pool import Pool

class GraphDataset(object):

    def __init__(self,folder_path,ordered=False):
        super().__init__()

        self.file_paths=glob.glob(f'{folder_path}/*.npz')
        self.file_paths.sort()
        self.ordered=ordered

        if self.ordered:
            self.i = 0

    def __len__(self):
        return len(self.file_paths)
    
    def get(self):
        if self.ordered:
            file_path = self.file_paths[self.i]
            self.i = (self.i + 1)%len(self.file_paths)
        else:
            file_path = random.sample(self.file_paths, k=1)[0]
        return load_npz(file_path).toarray()
    


@njit
def flatten_graph(graph):
    """
    Flatten a graph into matrices for adjacency, weights, start indices, and end indices.

    Parameters:
    - graph (adjacency matrix): The input graph to be flattened.

    Returns:
    - numpy.ndarray: Flattened adjacency matrix.
    - numpy.ndarray: Flattened weight matrix.
    - numpy.ndarray: Start indices for nodes in the flattened matrices.
    - numpy.ndarray: End indices for nodes in the flattened matrices.
    """
    flattened_adjacency = []
    flattened_weights = []
    num_nodes = graph.shape[0]
    
    node_start_indices = np.zeros(num_nodes,dtype=np.int64)
    node_end_indices = np.zeros(num_nodes,dtype=np.int64)
    
    for i in range(num_nodes):
        node_start_indices[i] = len(flattened_adjacency)
        for j in range(num_nodes):
            if graph[i, j] != 0:
                flattened_adjacency.append(j)
                flattened_weights.append(graph[i, j])
                
        node_end_indices[i] = len(flattened_adjacency)

    return (
        np.array(flattened_adjacency),
        np.array(flattened_weights),
        node_start_indices,
        node_end_indices
    )


@njit
def LS(graph,spins):
    adj_matrix, weight_matrix, start_list, end_list=graph
    
    n=len(start_list)
    delta_local_cuts=np.zeros(n)
    
    curr_score=0
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                  weight_matrix[start_list[i]:end_list[i]]):
                
            delta_local_cuts[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)
            curr_score+=weight*(spins[i]+spins[j]-2*spins[i]*spins[j])
            

    curr_score/=2    
    best_score=curr_score
    # print('Intital Best Score',curr_score)
    

    # for t in range(1000):
    while True:
        # arg_gain=np.argsort(-delta_local_cuts)
        arg_gain=np.argmax(delta_local_cuts)


        # add/delete
        
        best_add_remove_gain=max(delta_local_cuts[arg_gain],0)
    
        # swapping
        best_swap_gain=0

        element_1=-1
        element_2=-2
        for i in range(n):
            if spins[i]==0:
                delta_local_cuts_copy=np.copy(delta_local_cuts)
                spins_copy=np.copy(spins)
                new_score=curr_score+delta_local_cuts_copy[i]
                delta_local_cuts_copy[i]=-delta_local_cuts_copy[i]
                for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                                     weight_matrix[start_list[i]:end_list[i]]):

                    delta_local_cuts_copy[j]+=weight*(2*spins_copy[j]-1)*(2-4*spins_copy[i])

                spins_copy[i]=1-spins_copy[i]

                for j in range(n):
                    if spins[j]==1:
                        # continue
                        if new_score+delta_local_cuts_copy[j]-curr_score>best_swap_gain:
                            best_swap_gain=new_score+delta_local_cuts_copy[j]-curr_score
                            element_1=i
                            element_2=j

        
        if best_add_remove_gain==0 and best_swap_gain==0:
            break
        elif best_add_remove_gain>=best_swap_gain:
            # print('Adding')
            curr_score+=best_add_remove_gain
            delta_local_cuts[arg_gain]=-delta_local_cuts[arg_gain]
            for u,weight in zip(adj_matrix[start_list[arg_gain]:end_list[arg_gain]],
                                    weight_matrix[start_list[arg_gain]:end_list[arg_gain]]):

                delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[arg_gain])

            spins[arg_gain] = 1-spins[arg_gain]


        else:
            # print('swapping')
            curr_score+=best_swap_gain
            delta_local_cuts[element_1]=-delta_local_cuts[element_1]
            for u,weight in zip(adj_matrix[start_list[element_1]:end_list[element_1]],
                                    weight_matrix[start_list[element_1]:end_list[element_1]]):

                delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[element_1])

            spins[element_1] = 1-spins[element_1]

            delta_local_cuts[element_2]=-delta_local_cuts[element_2]
            for u,weight in zip(adj_matrix[start_list[element_2]:end_list[element_2]],
                                    weight_matrix[start_list[element_2]:end_list[element_2]]):

                delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[element_2])

            spins[element_2] = 1-spins[element_2]
        # print('Best add remove:',best_add_remove_gain)
        # print('Best add remove:',best_add_remove_gain)
        # print(best_swap_gain)
        # print('Curr_score',curr_score)
        best_score=max(curr_score,best_score)

    return best_score


from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--distribution", type=str,default='Physics', help="Distribution of dataset")
    parser.add_argument("--num_repeat", type=int,default=50, help="num_repeat")
    # parser.add_argument("--step_factor", type=int,default=2, help="Step factor")
    

    args = parser.parse_args()

    save_folder=f'pretrained agents/{args.distribution}_LS/data'
    os.makedirs(save_folder,exist_ok=True)

    test_dataset=GraphDataset(f'../data/testing/{args.distribution}',ordered=True)

    print("Number of test graphs:",len(test_dataset))
    best_cuts=[]
    for i in range(len(test_dataset)):
        graph=test_dataset.get()
        g=flatten_graph(graph)

        n=graph.shape[0]

        arguments=[]

        for i in range(args.num_repeat):
            spins= np.random.randint(2, size=graph.shape[0])
            arguments.append((g,spins))
            
        with Pool(40) as pool:
            best_cut=np.max(pool.starmap(LS, arguments))

        best_cuts.append(best_cut)

    best_cuts=np.array(best_cuts)

    df={'LS':best_cuts}
    df['Instance'] = [os.path.basename(file) for file in test_dataset.file_paths]
    df=pd.DataFrame(df)

    print(df)
    
    df.to_pickle(os.path.join(save_folder,f'results'))


            

        












