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


def get_train_test(rating_mat):
    n,m = rating_mat.shape
    
    selected_items,rest_ratings,negtive_items = [],[],[]
    for user_line in rating_mat:
        rated_items = np.where(user_line>0)[0]
        rated_num = len(rated_items)
        random_ids = [i for i in range(rated_num)]
        np.random.shuffle(random_ids)
        selected_id = random_ids[0]
        selected_items.append(rated_items[selected_id])
        rest_ratings.append(rated_items[random_ids[1:]])
        
        unrated_items = np.where(user_line==0)[0]
        unrated_num = len(unrated_items)
        random_ids = [i for i in range(unrated_num)]
        np.random.shuffle(random_ids)
        negtive_items.append(unrated_items[random_ids[:99]])
        
    train = [[user, item, rating_mat[user,item]] for user in range(n) for item in rest_ratings[user]]   
    test = [[user, selected_items[user]] for user in range(n)]
    
    length = int(n*0.1) # Users with the 10% lowest score data
    rated_size = np.sum(rating_mat>0,1)
    rated_order = np.argsort(rated_size)
    sparse_user = rated_order[:length]
    
    np.random.shuffle(train)  
    return train,test,negtive_items,sparse_user


def write_list(records_list, filename):
    file = "output/"  + str(filename) + ".txt"
    with open(file, "a", encoding='utf-8') as f:
        for line in records_list:
            f.write(','.join([str(i) for i in line]) + '\n') 
            
            
def read_data(file_dir):
    data=[]
    f = open(file_dir)
    for line in f.readlines():
        d, t = line.split(',')
        data.append([int(d), int(t)])
    
    return data


def generate_data(train_mat, sample_size=4, mode=0):
    drugs_num,targets_num = train_mat.shape
    data = []
    
    if mode==0:
        for d in range(drugs_num):
            positive_targets = np.where(train_mat[d,:]>0)[0] #drug 中大于零的项

            for target0 in positive_targets:
                data.append([d,target0,1])
                i = 0
                while i<sample_size:
                    target1 = np.random.randint(targets_num)
                    if (target1 not in positive_targets) and (train_mat[d,target1]==0):
                        data.append([d,target1,0])
                        i = i+1
    else:
        for d in range(drugs_num):
            positive_targets = np.where(train_mat[d,:]>0)[0] #drug 中大于零的项

            for target0 in positive_targets:
                i = 0
                while i<sample_size:
                    target1 = np.random.randint(targets_num)
                    if (target1 not in positive_targets) and (train_mat[d,target1]==0):
                        data.append([d,target0,target1])
                        i = i+1
    return data