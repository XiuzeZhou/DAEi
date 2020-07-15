import numpy as np

def load_data(file_dir):
    # output: 
    # N: the number of user;
    # M: the number of item
    # data: the list of rating information
    drug_ids_dict, target_ids_dict = {},{}
    N,M,d_idx,t_idx = 0,0,0,0 # N: the number of drug; M: the number of target
    data = []
    f = open(file_dir)
    for line in f.readlines():
        d, t = line.split()
        d = d.replace(':','')		
        if d not in drug_ids_dict:
            drug_ids_dict[d]=d_idx
            d_idx+=1
        if t not in target_ids_dict:
            target_ids_dict[t]=t_idx
            t_idx+=1
        data.append([drug_ids_dict[d],target_ids_dict[t],1])
    
    f.close()
    N = d_idx
    M = t_idx

    return N, M, data, drug_ids_dict, target_ids_dict


def sequence2mat(sequence, N, M):
    # input:
    # sequence: the list of rating information
    # N: row number, i.e. the number of users
    # M: column number, i.e. the number of items
    # output:
    # mat: user-item rating matrix
    records_array = np.array(sequence)
    mat = np.zeros([N,M])
    row = records_array[:,0].astype(int)
    col = records_array[:,1].astype(int)
    values = records_array[:,2].astype(np.float32)
    mat[row,col]=values
    
    return mat
