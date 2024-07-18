#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[1]:


import dill                            
filepath = 'session.pkl'


# In[2]:


dill.load_session(filepath)


# In[5]:


import time
import platform


# In[6]:


from scipy import optimize


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


from numba import jit


# In[9]:


Z = np.array([0.26061824, 0.3356399, 0.43225732, 0.55668708, 0.71693522, 0.92331246, 1.18909753, 1.53139158, 1.97221853, 2.53994208, 3.27109075])


# In[10]:


grid_number_Z = 11


# In[11]:


Pi = [
    0.776, 0.199, 0.023, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.020, 0.781, 0.180, 0.018, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.001, 0.040, 0.784, 0.160, 0.014, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.002, 0.060, 0.787, 0.140, 0.011, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.003, 0.080, 0.789, 0.120, 0.008, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.005, 0.100, 0.789, 0.100, 0.006, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.008, 0.120, 0.789, 0.080, 0.003, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.011, 0.140, 0.787, 0.060, 0.002, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.014, 0.160, 0.784, 0.040, 0.001,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.018, 0.180, 0.781, 0.020,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.002, 0.023, 0.199, 0.776
]

Pi = np.array(Pi).reshape(11, 11)


# In[12]:


#calibration
sigma = 2
beta = 0.964
alpha = 0.3
delta = 0.1


# 1a

# In[348]:


tau = 0.05
r = 0.029
k_upper = 100
k_lower = 0

grid_number_k = 250


# In[349]:


k_grid = np.linspace(k_lower, k_upper, grid_number_k)


# In[350]:


@jit
def K_generator(r):
    return ((r/(1-tau)+delta)/alpha)**(1/(alpha-1))


# In[351]:


@jit
def r_generator(K, tau):
    return (1-tau)*(alpha*K**(alpha-1)-delta)


# In[352]:


@jit
def wage(r, tau):
    K = ((r/(1-tau)+delta)/alpha)**(1/(alpha-1))
    return (1-alpha)*K**alpha


# In[353]:


@jit
def u(x):
    if x>0:
        return 1/(1-sigma)*x**(1-sigma)
    else:
        return -np.inf


# In[354]:


w = wage(r, tau)
print(w)


# In[355]:


tau


# In[356]:


@jit
def bellman_operator(V_old, r, w, Z, Pi, k_grid):
    
    V_new = np.zeros_like(V_old)
    k_pol = np.zeros_like(V_old)
    k_policy = np.zeros_like(V_old)
    
    for i_k, k in enumerate(k_grid):
        
        for i_z, z in enumerate(Z):
            
            Prob = Pi[i_z,:]
            
            V_compare = np.zeros_like(k_grid)
            
            for i_k_prime, k_prime in enumerate(k_grid):
                
                expectation = np.sum(Prob*V_old[i_k_prime, :])
                
                V_compare[i_k_prime] = u((1+r)*k+w*z-k_prime) + beta*expectation
            
            #what to do if all choices make your consumption negative
            
            if np.max(V_compare) == -np.inf:
                
                V_new[i_k, i_z] = -10000
                
                k_pol[i_k, i_z] = i_k
                
                k_policy[i_k, i_z] = k
            
            else:
                
                V_new[i_k, i_z] = np.max(V_compare)
                
                k_pol[i_k, i_z] = np.argmax(V_compare)
                
                k_policy[i_k, i_z] = k_grid[int(k_pol[i_k, i_z])]
                
    return V_new, k_pol, k_policy
                


# In[357]:


max_itr = 500
tol = 1e-5


# In[358]:


@jit
def bellman_iteration(V_init, r, w, Z, Pi, k_grid):
    
    error = 1
    
    itr = 1
    
    V_old = V_init
    
    while itr < max_itr and error > tol:
        
        V_new = bellman_operator(V_old, r, w, Z, Pi, k_grid)[0]
        
        error = np.max(np.abs(V_new - V_old))
        
        V_old = V_new
        
        itr += 1 
        
        
    return bellman_operator(V_old, r, w, Z, Pi, k_grid)    


# In[359]:


V_init = np.zeros((grid_number_k, grid_number_Z))


# In[360]:


V, k_pol, k_policy = bellman_iteration(V_init, r, w, Z, Pi, k_grid)


# In[454]:


plt.plot(k_grid, k_policy[:, 2], color = "y", label = "$z = z_3$")
plt.plot(k_grid, k_policy[:, 5], color = "r", label = "$z = z_6$")
plt.plot(k_grid, k_policy[:, 8], color = "b", label = "$z = z_9$")
plt.legend()
plt.xlabel("k")
plt.ylabel("k'")
plt.savefig("1a1.pdf")


# In[362]:


plt.plot(k_grid, V[:, 2], color = "y")
plt.plot(k_grid, V[:, 5], color = "r")
plt.plot(k_grid, V[:, 8], color = "b")


# 1b

# In[363]:


#first compute the invariant distribution of the labor supply


# In[364]:


@jit
def stationary_distribution(Pi):
    dist_temp = np.ones(grid_number_Z)/grid_number_Z
    
    dist_old = dist_temp
    
    error = 1
    
    while error > tol:
        
        dist_new = np.matmul(dist_old, Pi)
        
        error = np.max(np.abs(dist_new - dist_old))
        
        dist_old = dist_new
    
    return dist_old


# In[366]:


Z_dist_stat = stationary_distribution(Pi)


# In[367]:


@jit 
def next_state_dist_non(curr_dist, k_grid, transitional_matrix, k_policy, k_pol):
    
    next_dist = np.zeros_like(curr_dist)
    
    len_trans = len(transitional_matrix)
    
    len_state = len(k_grid)
    
    indicator_mat = np.zeros((len_state, len_trans, len_state))
    
    for i in range(len_state):
        for j in range(len_trans):
            
            k_pol_temp = k_pol[i,j]
            
            indicator_mat[i,j, int(k_pol_temp)] = 1
            
    for i_prime in range(len_state):
        
        for j_prime in range(len_trans):
            
            for i in range(len_state):
                
                for j in range(len_trans):
                    
                    next_dist[i_prime, j_prime] += curr_dist[i,j]*indicator_mat[i,j,i_prime]*transitional_matrix[j,j_prime]
                    
    return  next_dist           


# In[368]:


@jit
def next_state_dist(curr_dist, k_grid_b, transitional_matrix, k_policy):
    
    next_dist = np.zeros_like(curr_dist)
    
    len_trans = len(transitional_matrix)
    
    len_state = len(k_grid_b)
    
    len_ind = len(k_grid_b)-1
    
    indicator_mat = np.zeros((len_state, len_trans, len_ind))
    
    policy_temp = k_policy
    
    #indicator matrix
    for i in range(len_state):
        for j in range(len_trans):
            
            k_policy_temp = np.interp(k_grid_b[i], k_grid, policy_temp[:, j])
            
            if k_grid_b[-2] <= k_policy_temp <= k_grid_b[-1]:
                indicator_mat[i,j,len_ind-1] = 1
            else:
                indicator_mat[i,j,len_ind-1] = 0
            
            for k in range(len_ind-1):
                
                if k_grid_b[k] <= k_policy_temp < k_grid_b[k+1]:
                    indicator_mat[i,j,k] = 1
                else:
                    indicator_mat[i,j,k] = 0
            
    
    #initial one 
    for i in range(len_trans):
        for j in range(len_trans):
            for m in range(len_state):
                
                k_policy_temp = np.interp(k_grid_b[m], k_grid, policy_temp[:, j])
                
                next_dist[0,i] += transitional_matrix[j,i]*(k_grid_b[1]-k_policy_temp)/(k_grid_b[1]-k_grid_b[0])*indicator_mat[m,j,0]*curr_dist[m, j]
    #last one
    for i in range(len_trans):
        for j in range(len_trans):
            for m in range(len_state):
                
                k_policy_temp = np.interp(k_grid_b[m], k_grid, policy_temp[:, j])
                
                next_dist[-1, i] += transitional_matrix[j,i]*(k_policy_temp-k_grid_b[-2])/(k_grid_b[-1]-k_grid_b[-2])*indicator_mat[m, j, len_ind-1]*curr_dist[m, j]
    
    #between these
    for l in range(1, len_state-1):
        for i in range(len_trans):
            for j in range(len_trans):
                for m in range(len_state):
                
                    k_policy_temp = np.interp(k_grid_b[m], k_grid, policy_temp[:, j])
                    
                    next_dist[l,i] += transitional_matrix[j,i]*(k_policy_temp-k_grid_b[l-1])/(k_grid_b[l]-k_grid_b[l-1])*indicator_mat[m,j,l-1]*curr_dist[m, j]+\
                                      transitional_matrix[j,i]*(k_grid_b[l+1]-k_policy_temp)/(k_grid_b[l+1]-k_grid_b[l])*indicator_mat[m,j,l]*curr_dist[m, j]
                                    
                                        
    return next_dist


# In[369]:


grid_number_k_b = 250


# In[370]:


k_grid_b = np.linspace(k_lower, k_upper, grid_number_k_b)


# In[371]:


dist_init = np.zeros((grid_number_k_b, grid_number_Z))
for i in range(grid_number_k_b):
    for j in range(grid_number_Z):
        dist_init[i,j] = Z_dist_stat[j]/grid_number_k_b


# In[372]:


dist_trial = next_state_dist(dist_init, k_grid_b, Pi, k_policy)


# In[374]:


np.sum(dist_trial)


# In[375]:


@jit
def stat_dist_finder(dist_init, k_grid_b, transitional_matrix, k_policy, k_pol):
    
    error = 1
    itr = 1
    
    dist_old = dist_init
    
    while error>tol and itr<max_itr:
        
        dist_new = next_state_dist_non(dist_old, k_grid_b, transitional_matrix, k_policy, k_pol)
        
        dist_new = dist_new/np.sum(dist_new)
        
        error = np.max(np.abs(dist_new-dist_old))
        
        dist_old = dist_new
        
        itr += 1
        
    return dist_old


# In[376]:


tol = 1e-5


# In[377]:


max_itr = 1000


# In[378]:


dist_stat = stat_dist_finder(dist_init, k_grid, Pi, k_policy, k_pol)


# In[379]:


k_stat = np.zeros(grid_number_k_b)
for i in range(grid_number_k_b):
    k_stat[i] = np.sum(dist_stat[i,:])


# In[380]:


np.sum(k_stat)


# In[381]:


K_temp = np.sum(k_grid_b*k_stat)


# In[382]:


K_temp


# In[455]:


plt.bar(k_grid_b, k_stat, color = "maroon")
plt.xlabel("k")
plt.ylabel("frequency")
plt.savefig("1b1.pdf")


# In[384]:


diff_1 = r_generator(K_temp, tau)-r
print(diff_1)


# I first compute the difference in interest rate for r = 0.040 and r = 0.050. For r = 0.050, interest rate diff is negative, while for r = 0.040, the interest rate diff is positive. Choose the gap between jump to be 0.001, and we will write a function to find the root.

# In[101]:


@jit 
def diff_generator(tau, k_grid, r, dist_init, Pi):
    
    w = wage(r, tau)
    
    V_init = np.zeros((grid_number_k, grid_number_Z))
    
    V, k_pol, k_policy = bellman_iteration(V_init, r, w, Z, Pi, k_grid)
    
    dist_stat = stat_dist_finder(dist_init, k_grid, Pi, k_policy, k_pol)
    
    k_stat = np.zeros(grid_number_k_b)
    for i in range(grid_number_k_b):
        k_stat[i] = np.sum(dist_stat[i,:])
        
    K_temp = np.sum(k_grid*k_stat)
    
    return r_generator(K_temp, tau)-r


# In[55]:


grid_number_r = 11


# In[56]:


r_grid = np.linspace(0.040, 0.050, grid_number_r)


# In[133]:


diff_generator(tau, k_grid, 0.02125, dist_init, Pi)


# In[165]:


diff_array = np.zeros(grid_number_r)
for i in range(grid_number_r):
    diff_array[i] = diff_generator(tau, k_grid, r_grid[i], dist_init, Pi)


# In[166]:


diff_array


# In[142]:


diff_generator(tau, k_grid, 0.0211945, dist_init, Pi)


# In[189]:


g = lambda r: diff_generator(tau, k_grid, r, dist_init, Pi)


# In[191]:


r_trial = optimize.brentq(g, 0.049, 0.05)


# In[192]:


#with a further revision, we find that r = 0.0213
r_opt = r_trial


# In[399]:


r_opt = 0.0211945


# In[400]:


w_opt = wage(r_opt, tau)
    
V_init = np.zeros((grid_number_k, grid_number_Z))
    
V_opt, k_pol_opt, k_policy_opt = bellman_iteration(V_init, r_opt, w_opt, Z, Pi, k_grid)
    
dist_stat_opt = stat_dist_finder(dist_init, k_grid, Pi, k_policy_opt, k_pol_opt)
    
k_stat_opt = np.zeros(grid_number_k_b)

for i in range(grid_number_k_b):
    
    k_stat_opt[i] = np.sum(dist_stat_opt[i,:])


# In[401]:


dist_stat_opt


# In[402]:


plt.plot(k_grid, k_policy_opt[:, 2], color = "y")
plt.plot(k_grid, k_policy_opt[:, 5], color = "r")
plt.plot(k_grid, k_policy_opt[:, 8], color = "b")


# In[403]:


np.sum(dist_init)


# In[404]:


K_temp_opt = np.sum(k_grid*k_stat_opt)


# In[405]:


print(w_opt)
print(K_temp_opt)


# In[406]:


K_generator(r_opt)


# In[457]:


Y = K_temp_opt**alpha
print(Y)


# In[408]:


c_matrix = np.zeros((grid_number_k, grid_number_Z))
C = 0

for i in range(grid_number_k):
    for j in range(grid_number_Z):
        c_matrix[i,j] = (1+r_opt)*k_grid[i]+w_opt*Z[j]-k_grid[int(k_pol_opt[i,j])]
        C += c_matrix[i,j]*dist_stat_opt[i,j]


# In[409]:


c_matrix


# In[410]:


print(C)


# In[463]:


(1.0315-1.0282)/(1.0282)


# ## 2 An Economic Boom

# In[204]:


pip install networkx


# In[159]:


import networkx as nx


# In[160]:


# Setup the figure and axis
fig, ax = plt.subplots(figsize=(12, 10))

# Create a directed graph
G = nx.DiGraph()

# Add nodes with the label as node name
nodes = {
    "unknown K\nshock τ": (-1, -1),
    "het.agent": (1, 0.5),
    "rep.firm": (0, 0.5),
    "goods mkt. \n clearing\n $H = K^s - K$": (2, -1)
}

# Add edges with labels
edges = {
    ("unknown K\nshock τ", "rep.firm"): "K, τ",
    ("rep.firm", "het.agent"): "r, w",
    ("het.agent", "goods mkt. \n clearing\n $H = K^s - K$"): "$K^s$",
    ("unknown K\nshock τ", "goods mkt. \n clearing\n $H = K^s - K$"): "$K$"
}

# Add nodes to the graph
for node, (x, y) in nodes.items():
    G.add_node(node, pos=(x, y))

# Add edges to the graph
for edge, label in edges.items():
    G.add_edge(*edge, label=label)

# Get positions
pos = nx.get_node_attributes(G, 'pos')

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=7000, node_color='white', edgecolors='black')

# Draw edges with arrows, adjust connection style to shorten the edges
nx.draw_networkx_edges(
    G, pos, edgelist=edges.keys(), arrows=True,
    arrowstyle='-|>', arrowsize=20, edge_color='black',
    connectionstyle='arc3,rad=0.1',
    node_size=7000  # The node_size here should match the size used in draw_networkx_nodes
)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Edge labels
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', label_pos=0.5)

# Remove axis
plt.axis('off')
plt.savefig("DAG.pdf")
# Show the graph
plt.show()


# We have 
# 
# \begin{equation}
# \begin{aligned}
# r &= (1-\tau)(\alpha K^{\alpha-1}-\delta)\\
# w &= (1-\alpha)K^{\alpha}
# \end{aligned}
# \end{equation}
# 
# Therefore, we have
# \begin{equation}
# \begin{aligned}
# \frac{\partial r}{\partial K} &= (1-\tau)(\alpha K^{\alpha-1}-\delta)\\
# w &= (1-\alpha)K^{\alpha}
# \end{aligned}
# \end{equation}

# In[161]:


T = 50


# In[162]:


tau_ss = 0.05
tau_td = 0.00


# In[587]:


K_ss = K_temp_eg


# In[588]:


#simple block for firms


# In[755]:


#Jacobian of r with regard to K: diagonal
Jacobian_r_K = np.eye(T)*(1-tau)*alpha*(alpha-1)*K_ss**(alpha-2)


# In[763]:


(1-tau)*alpha*(alpha-1)*K_ss**(alpha-2)


# In[756]:


Jacobian_r_K


# In[757]:


#Jacobian of r with regard to tau: diagonal
Jacobian_r_tau = np.eye(T)*(-(alpha*K_ss**(alpha-1)-delta))
print(Jacobian_r_tau)


# In[766]:


#Jacobian of w with regard to K: diagonal
Jacobian_w_K = np.eye(T)*((1-alpha)*alpha*K_ss**(alpha-1))

#Jacobian of w with regard to tau: diagonal
Jacobian_w_tau = np.eye(T)*(0)


# In[767]:


Jacobian_w_K


# In[171]:


#we need to compute K_s w.r.t. r and K_s w.r.t. w
#we follow the fake news algorithm
#first we write the function that backs out policy function and transitional matrix given 


# In[ ]:





# In[ ]:





# In[185]:


def du(c):
    if c>0:
        return c**(-sigma)
    else:
        return +10000


# In[186]:


def consumption(r, k, z, w, k_prime):
    return  (1+r)*k + w*z - k_prime


# In[187]:


def u_prime_inverse(x):
    if x>0:
        return x**(-1/sigma)
    else: 
        return +np.inf


# In[188]:


#to maintain the sensitivity of policy function to change in tau as much as possible, I recalculate everything using policy function
#iteration with endogeneous grid method

def policy_operator_eg(k_policy, w, r, k_grid, Z, Pi):
    
    gamma = (1+r)*beta
    
    #update policy function
    k_policy_updated = np.empty(k_policy.shape)
    
    for i_z, z in enumerate(Z):
        
        k_eg = np.zeros(grid_number_k)
        
        Prob = Pi[i_z, :]
        
        for i_k, k in enumerate(k_grid):
            
            expectation = 0
            
            for j in range(len(Z)):
                
                expectation = expectation + du(consumption(r, k, Z[j], w, k_policy[i_k, j]))*Prob[j]
                
            k_eg[i_k] = (u_prime_inverse(gamma*expectation) + k - w*z)/(1+r)
            
        for i_k in range(len(k_grid)):

            k_policy_updated[i_k, i_z] = np.interp(k_grid[i_k], k_eg, k_grid)
            
    return k_policy_updated#,k_grid[-1]*np.ones((grid_number_k,grid_number_Z)))


# In[189]:


k_init = np.zeros((grid_number_k, grid_number_Z))
for i in range(grid_number_k):
    for j in range(grid_number_Z):
        k_init[i,j] = k_grid[i]


# In[291]:


tol = 10e-9


# In[292]:


def policy_iteration(k_init, w, r, k_grid, Z, Pi):
    
    k_old = k_init
    error = 1
    itr = 1
    
    while error > tol and itr < max_itr:
        itr += 1
        k_update = policy_operator_eg(k_old, w, r, k_grid, Z, Pi)
        error = np.max(np.abs(k_old-k_update))
        k_old = k_update
        
    return policy_operator_eg(k_old, w, r, k_grid, Z, Pi)


# In[293]:


r_eg = 0.02042

w_eg = wage(r_eg, tau)


# In[294]:


0.0211945


# In[295]:


print('system:',platform.system())
T1 = time.perf_counter()
k_policy_eg = policy_iteration(k_init, w_eg, r_eg, k_grid, Z, Pi)
T2 = time.perf_counter()
print('Time:%sms' % ((T2 - T1)*1000))


# In[296]:


@jit
def stat_dist_finder_eg(dist_init, k_grid_b, transitional_matrix, k_policy):
    
    error = 1
    itr = 1
    
    dist_old = dist_init
    
    while error>tol and itr<max_itr:
        
        dist_new = next_state_dist(dist_old, k_grid_b, transitional_matrix, k_policy)
        
        dist_new = dist_new/np.sum(dist_new)
        
        error = np.max(np.abs(dist_new-dist_old))
        
        dist_old = dist_new
        
        print(error)
        
        itr += 1
        
    return dist_old


# In[483]:


dist_stat_eg = stat_dist_finder_eg(dist_stat_opt, k_grid, Pi, k_policy_eg)


# In[298]:


k_stat_eg = np.zeros(grid_number_k)
for i in range(grid_number_k):
    k_stat_eg[i] = np.sum(dist_stat_eg[i,:])


# In[299]:


K_temp_eg = np.sum(k_grid*k_stat_eg)
print(K_temp_eg)
K_generator(r_eg)


# In[385]:


C_temp_eg = 0
for i in range(grid_number_k):
    for j in range(grid_number_Z):
        C_temp_eg += c_policy_eg[i,j]*dist_stat_eg[i,j]


# In[386]:


C_temp_eg


# In[300]:


#above is only endogenous grid method, which turns out to be coverging so slowly when trying to find stationary distribution
#this is because grid does not match and has to run pdf iteration to update distribution; let's stick with the original stationary distribution wen have
#but when doing backward iteration, we use endogenous grid method to get the points


# In[301]:


#first compute Jacobian of K_s to w


# In[724]:


@jit
def transitional_matrix_generator(k_policy, k_grid):
    
    y_temp = k_policy.reshape((1,-1))
    
    #stack state variables
    n_g = len(y_temp[0])
    
    Lambda_temp = np.zeros((n_g, n_g))
    
    for i_g in range(n_g):
        
        i_k = i_g//grid_number_Z
        
        i_z = i_g%grid_number_Z
        
        policy_temp = k_policy[i_k, i_z]
        
        for i_k in range(grid_number_k-1):
            
            if k_grid[i_k]<=policy_temp<k_grid[i_k+1]:
                
                ind = i_k
                
                break
                
        for i_z_prime in range(grid_number_Z):
            
            Lambda_temp[i_g, ind*grid_number_Z+i_z_prime] += (k_grid[ind+1]-policy_temp)/(k_grid[ind+1]-k_grid[ind])*Pi[i_z, i_z_prime]
            
            Lambda_temp[i_g, (ind+1)*grid_number_Z+i_z_prime] += (policy_temp-k_grid[ind])/(k_grid[ind+1]-k_grid[ind])*Pi[i_z, i_z_prime]
            
    return Lambda_temp


# In[57]:


@jit
def c_policy_generator(k_policy, w, r, k_grid, Z, Pi):
    
    c_policy = np.zeros_like(k_policy)
    
    for i_k, k in enumerate(k_grid):
        
        for i_z, z in enumerate(Z):
            
            c_policy[i_k, i_z] = w*z+(1+r)*k - k_policy[i_k, i_z]
            
    return c_policy


# In[262]:


c_policy_eg = c_policy_generator(k_policy_eg, w_eg, r_eg, k_grid, Z, Pi)


# In[263]:


dw = 0.001*w_eg


# In[281]:


w_T1_plus = w_eg + dw
r_T1_plus = r_eg 

k_policy_T1_plus = policy_operator_eg(k_policy_eg, w_T1_plus, r_T1_plus, k_grid, Z, Pi)

c_policy_T1_plus = c_policy_generator(k_policy_T1_plus, w_T1_plus, r_T1_plus, k_grid, Z, Pi)

policy_k_plus = {}

policy_k_plus["T1"] = k_policy_T1_plus

policy_c_plus = {}

policy_c_plus["T1"] = c_policy_T1_plus

for i in range(1,T):
    
    policy_k_plus["T"+str(i+1)] = policy_operator_eg(policy_k_plus["T"+str(i)], w_eg, r_eg, k_grid, Z, Pi)
    
    policy_c_plus["T"+str(i+1)] = c_policy_generator(policy_k_plus["T"+str(i+1)], w_eg, r_eg, k_grid, Z, Pi)
    
trans_mat_plus = {}

for i in range(T):
    
    trans_mat_plus["T"+str(i+1)] = transitional_matrix_generator(policy_k_plus["T"+str(i+1)], k_grid)


# In[282]:


w_T1_minus = w_eg - dw
r_T1_minus = r_eg 

k_policy_T1_minus = policy_operator_eg(k_policy_eg, w_T1_minus, r_T1_minus, k_grid, Z, Pi)

c_policy_T1_minus = c_policy_generator(k_policy_T1_minus, w_T1_minus, r_T1_minus, k_grid, Z, Pi)

policy_k_minus = {}

policy_k_minus["T1"] = k_policy_T1_minus

policy_c_minus = {}

policy_c_minus["T1"] = c_policy_T1_minus

for i in range(1,T):
    
    policy_k_minus["T"+str(i+1)] = policy_operator_eg(policy_k_minus["T"+str(i)], w_eg, r_eg, k_grid, Z, Pi)
    
    policy_c_minus["T"+str(i+1)] = c_policy_generator(policy_k_minus["T"+str(i+1)], w_eg, r_eg, k_grid, Z, Pi)
    
trans_mat_minus = {}

for i in range(T):
    
    trans_mat_minus["T"+str(i+1)] = transitional_matrix_generator(policy_k_minus["T"+str(i+1)], k_grid)


# In[283]:


#now lets calculate the Jacobian


# In[267]:


dist_stat_eg_stack = dist_stat_eg.reshape((1,-1))


# In[268]:


Lambda_ss = transitional_matrix_generator(k_policy_eg, k_grid)


# In[269]:


gamma_k_w = np.zeros(T)
gamma_c_w = np.zeros(T)


# In[270]:


for i in range(T):
    gamma_k_w[i] = (np.sum(policy_k_plus["T"+str(i+1)].reshape((1,-1))*dist_stat_eg_stack)-np.sum(policy_k_minus["T"+str(i+1)].reshape((1,-1))*dist_stat_eg_stack))/(2*dw)


# In[287]:


policy_c_plus["T"+str(2)] - policy_c_minus["T"+str(2)]


# In[271]:


for i in range(T):
    gamma_c_w[i] = (np.sum(policy_c_plus["T"+str(i+1)].reshape((1,-1))*dist_stat_eg_stack)-np.sum(policy_c_minus["T"+str(i+1)].reshape((1,-1))*dist_stat_eg_stack))/(2*dw)


# In[272]:


D_k_w = {}
D_c_w = {}


# In[273]:


for i in range(T):
    
    dLambda = trans_mat_plus["T"+str(i+1)] - trans_mat_minus["T"+str(i+1)]
    
    D_s = np.matmul(np.transpose(dLambda), dist_stat_eg_stack[0])/(2*dw)
    
    D_k_w["T"+str(i)] = D_s
    D_c_w["T"+str(i)] = D_s


# In[274]:


Epsilon_k_w = {} 
Epsilon_c_w = {} 


# In[275]:


Epsilon_k_w["T"+str(0)] = k_policy_eg.reshape((1,-1))[0]
Epsilon_c_w["T"+str(0)] = c_policy_eg.reshape((1,-1))[0]
for i in range(1,T-1):
    Epsilon_k_w["T"+str(i)] = np.matmul(Lambda_ss,Epsilon_k_w["T"+str(i-1)])
    Epsilon_c_w["T"+str(i)] = np.matmul(Lambda_ss,Epsilon_c_w["T"+str(i-1)])


# In[276]:


F_k_w = np.zeros((T,T))
F_c_w = np.zeros((T,T))


# In[277]:


for j in range(T):
    F_k_w[0,j] = gamma_k_w[j]
    F_c_w[0,j] = gamma_c_w[j]
    
for i in range(1,T):
    ep_temp = np.transpose(Epsilon_k_w["T"+str(i-1)])
    ep_temp_c = np.transpose(Epsilon_c_w["T"+str(i-1)])
    for j in range(T):
        F_k_w[i,j] = ep_temp@D_k_w["T"+str(j)]
        F_c_w[i,j] = ep_temp_c@D_c_w["T"+str(j)]


# In[278]:


Jacobian_Ks_w = np.zeros((T,T))
for i in range(T):
    for j in range(T):
        ind = np.minimum(i,j)
        for j_1 in range(ind+1):
            Jacobian_Ks_w[i,j] += F_k_w[i-j_1, j-j_1]


# In[7]:


Jacobian_Ks_w


# In[279]:


Jacobian_c_w = np.zeros((T,T))
for i in range(T):
    for j in range(T):
        ind = np.minimum(i,j)
        for j_1 in range(ind+1):
            Jacobian_c_w[i,j] += F_c_w[i-j_1, j-j_1]


# In[280]:


Jacobian_c_w


# In[171]:


#now compute the Jacobian of K^s with regard to r


# In[412]:


dr = 0.001*r_eg


# In[413]:


w_T1_plus = w_eg 
r_T1_plus = r_eg + dr 

k_policy_T1_plus = policy_operator_eg(k_policy_eg, w_T1_plus, r_T1_plus, k_grid, Z, Pi)

c_policy_T1_plus = c_policy_generator(k_policy_T1_plus, w_T1_plus, r_T1_plus, k_grid, Z, Pi)

policy_k_plus = {}

policy_k_plus["T1"] = k_policy_T1_plus

policy_c_plus = {}

policy_c_plus["T1"] = c_policy_T1_plus

for i in range(1,T):
    
    policy_k_plus["T"+str(i+1)] = policy_operator_eg(policy_k_plus["T"+str(i)], w_eg, r_eg, k_grid, Z, Pi)
    
    policy_c_plus["T"+str(i+1)] = c_policy_generator(policy_k_plus["T"+str(i+1)], w_eg, r_eg, k_grid, Z, Pi)
    
trans_mat_plus = {}

for i in range(T):
    
    trans_mat_plus["T"+str(i+1)] = transitional_matrix_generator(policy_k_plus["T"+str(i+1)], k_grid)


# In[414]:


w_T1_minus = w_eg 
r_T1_minus = r_eg - dr

k_policy_T1_minus = policy_operator_eg(k_policy_eg, w_T1_minus, r_T1_minus, k_grid, Z, Pi)

c_policy_T1_minus = c_policy_generator(k_policy_T1_minus, w_T1_minus, r_T1_minus, k_grid, Z, Pi)

policy_k_minus = {}

policy_k_minus["T1"] = k_policy_T1_minus

policy_c_minus = {}

policy_c_minus["T1"] = c_policy_T1_minus

for i in range(1,T):
    
    policy_k_minus["T"+str(i+1)] = policy_operator_eg(policy_k_minus["T"+str(i)], w_eg, r_eg, k_grid, Z, Pi)
    
    policy_c_minus["T"+str(i+1)] = c_policy_generator(policy_k_minus["T"+str(i+1)], w_eg, r_eg, k_grid, Z, Pi)
    
trans_mat_minus = {}

for i in range(T):
    
    trans_mat_minus["T"+str(i+1)] = transitional_matrix_generator(policy_k_minus["T"+str(i+1)], k_grid)


# In[417]:


gamma_k_r = np.zeros(T)
gamma_c_r = np.zeros(T)


# In[418]:


for i in range(T):
    gamma_k_r[i] = (np.sum(policy_k_plus["T"+str(i+1)].reshape((1,-1))*dist_stat_eg_stack)-np.sum(policy_k_minus["T"+str(i+1)].reshape((1,-1))*dist_stat_eg_stack))/(2*dr)


# In[419]:


for i in range(T):
    gamma_c_r[i] = (np.sum(policy_c_plus["T"+str(i+1)].reshape((1,-1))*dist_stat_eg_stack)-np.sum(policy_c_minus["T"+str(i+1)].reshape((1,-1))*dist_stat_eg_stack))/(2*dr)


# In[420]:


D_k_r = {}
D_c_r = {}


# In[421]:


for i in range(T):
    
    dLambda = trans_mat_plus["T"+str(i+1)] - trans_mat_minus["T"+str(i+1)]
    
    D_s = np.matmul(np.transpose(dLambda), dist_stat_eg_stack[0])/(2*dr)
    
    D_k_r["T"+str(i)] = D_s
    D_c_r["T"+str(i)] = D_s


# In[422]:


Epsilon_k_r = {} 
Epsilon_c_r = {} 


# In[423]:


Epsilon_k_r["T"+str(0)] = k_policy_eg.reshape((1,-1))[0]
Epsilon_c_r["T"+str(0)] = c_policy_eg.reshape((1,-1))[0]
for i in range(1,T-1):
    Epsilon_k_r["T"+str(i)] = np.matmul(Lambda_ss,Epsilon_k_r["T"+str(i-1)])
    Epsilon_c_r["T"+str(i)] = np.matmul(Lambda_ss,Epsilon_c_r["T"+str(i-1)])


# In[424]:


F_k_r = np.zeros((T,T))
F_c_r = np.zeros((T,T))


# In[425]:


for j in range(T):
    F_k_r[0,j] = gamma_k_r[j]
    F_c_r[0,j] = gamma_c_r[j]
    
for i in range(1,T):
    ep_temp = np.transpose(Epsilon_k_r["T"+str(i-1)])
    ep_temp_c = np.transpose(Epsilon_c_r["T"+str(i-1)])
    for j in range(T):
        F_k_r[i,j] = ep_temp@D_k_r["T"+str(j)]
        F_c_r[i,j] = ep_temp_c@D_c_r["T"+str(j)]


# In[5]:


Jacobian_Ks_r = np.zeros((T,T))
for i in range(T):
    for j in range(T):
        ind = np.minimum(i,j)
        for j_1 in range(ind+1):
            Jacobian_Ks_r[i,j] += F_k_r[i-j_1, j-j_1]


# In[6]:


Jacobian_Ks_r


# In[427]:


Jacobian_c_r = np.zeros((T,T))
for i in range(T):
    for j in range(T):
        ind = np.minimum(i,j)
        for j_1 in range(ind+1):
            Jacobian_c_r[i,j] += F_c_r[i-j_1, j-j_1]


# d

# We have 
# \begin{equation*}
# \begin{aligned}
# \mathrm{d}K = - \left(\frac{\partial H}{\partial K}\right)^{-1}\left(\frac{\partial H}{\partial \tau}\right)\mathrm{d} \tau
# \end{aligned}
# \end{equation*}
# and, correspondingly,
# 
# \begin{equation*}
# \begin{aligned}
# \frac{\partial H}{\partial K} &= J^{K^s,r}J^{r,K} + J^{K^s,w}J^{w,K} - 1\\
# \frac{\partial H}{\partial \tau} &= J^{K^s,r}J^{r,\tau} + J^{K^s,w}J^{w,\tau}
# \end{aligned}
# \end{equation*}

# In[240]:


H_K = Jacobian_Ks_r@Jacobian_r_K + Jacobian_Ks_w@Jacobian_w_K - np.eye(T) 


# In[241]:


H_tau = Jacobian_Ks_r@Jacobian_r_tau + Jacobian_Ks_w@Jacobian_w_tau


# In[242]:


linear_ope = - np.linalg.inv(H_K)@H_tau


# In[243]:


dtau_array_1 = 0*np.ones(T)
for i in range(4):
    dtau_array_1[i] = -tau


# In[244]:


dtau_array_2 = 0*np.ones(T)
for i in range(8):
    dtau_array_2[i] = -tau


# In[245]:


G = {}


# In[246]:


G["K"] = linear_ope


# In[247]:


dK_array_1 = G["K"]@dtau_array_1
dK_array_1 = np.insert(dK_array_1, 0, 0)


# In[248]:


dK_array_2 = G["K"]@dtau_array_2
dK_array_2 = np.insert(dK_array_2, 0, 0)


# In[249]:


T_array = np.arange(T+1)


# In[468]:


plt.plot(T_array, dK_array_1/K_temp_eg*100, color = "r", label = "4 periods")
plt.plot(T_array, dK_array_2/K_temp_eg*100, color = "g", label = "8 periods")
plt.xlabel("periods")
plt.ylabel("percent deviation from ss (K)")
plt.legend()
plt.savefig("2d1.pdf")


# From representative firm block, we have 
# 
# \begin{equation*}
# \begin{aligned}
# \frac{\mathrm{d}r}{\mathrm{d}\tau} = J^{r,K}\frac{\mathrm{d}K}{\mathrm{d}\tau} + J^{r,\tau}
# \end{aligned}
# \end{equation*}

# In[21]:


G["r"] = Jacobian_r_K@G["K"] + Jacobian_r_tau


# In[35]:


dr_array_1 = G["r"]@dtau_array_1
r_array_1 = np.insert(dr_array_1, 0, 0)+r_eg


# In[37]:


dr_array_2 = G["r"]@dtau_array_2
r_array_2 = np.insert(dr_array_2, 0, 0)+r_eg


# In[476]:


plt.plot(T_array, (r_array_1-r_eg)/r_eg*100, color = "r", label = "4 periods")
plt.plot(T_array, (r_array_2-r_eg)/r_eg*100, color = "g", label = "8 periods")
plt.xlabel("periods")
plt.ylabel("percent deviation from ss")
plt.legend()
plt.savefig("2e3.pdf")


# For $w$, we have
# 
# \begin{equation*}
# \begin{aligned}
# w = (1-\alpha)K^{\alpha}
# \end{aligned}
# \end{equation*}
# which leads to 
# 
# \begin{equation*}
# \begin{aligned}
# \mathrm{d}w = (1-\alpha)\alpha K^{\alpha-1}\mathrm{d}K
# \end{aligned}
# \end{equation*}

# In[480]:


G["w"] = (1-alpha)*alpha*K_ss**(alpha-1)*G["K"]


# In[481]:


dw_array_1 = G["w"]@dtau_array_1
w_array_1 = np.insert(dw_array_1, 0, 0) + w_eg

dw_array_2 = G["w"]@dtau_array_2
w_array_2 = np.insert(dw_array_2, 0, 0) + w_eg


# In[482]:


plt.plot(T_array, (w_array_1-w_eg)/w_eg*100, color = "r", label = "4 periods")
plt.plot(T_array, (w_array_2-w_eg)/w_eg*100, color = "g", label = "8 periods")
plt.xlabel("periods")
plt.ylabel("percent deviation from ss")
plt.legend()
plt.savefig("2e4.pdf")


# For $Y$, we have
# 
# \begin{equation*}
# \begin{aligned}
# Y = K^{\alpha}
# \end{aligned}
# \end{equation*}
# which leads to 
# 
# \begin{equation*}
# \begin{aligned}
# \mathrm{d}Y = \alpha K^{\alpha-1}\mathrm{d}K
# \end{aligned}
# \end{equation*}

# In[434]:


G["Y"] = alpha*K_ss**(alpha-1)*G["K"]


# In[477]:


dY_array_1 = G["Y"]@dtau_array_1
dY_array_1 = np.insert(dY_array_1, 0, 0)

dY_array_2 = G["Y"]@dtau_array_2
dY_array_2 = np.insert(dY_array_2, 0, 0)


# In[478]:


Y_eg = K_temp_eg**(alpha)


# In[479]:


plt.plot(T_array, (dY_array_1/Y_eg)*100, color = "r", label = "4 periods")
plt.plot(T_array, (dY_array_2/Y_eg)*100, color = "g", label = "8 periods")
plt.xlabel("periods")
plt.ylabel("deviation from ss")
plt.legend()
plt.savefig("2e2.pdf")


# For c, I recalculate the Jacobian using het.agent block. 

# In[253]:


G["C"] = Jacobian_c_w@(Jacobian_w_tau+Jacobian_w_K@G["K"]) + Jacobian_c_r@(Jacobian_r_tau+Jacobian_r_K@G["K"])


# In[293]:


Jacobian_c_r


# In[448]:


dC_array_1 = G["C"]@dtau_array_1
dC_array_1 = np.insert(dC_array_1, -1, 0)

dC_array_2 = G["C"]@dtau_array_2
dC_array_2 = np.insert(dC_array_2, -1, 0)


# In[470]:


plt.plot(T_array, dC_array_1/C_temp_eg*100, color = "r", label = "4 periods")
plt.plot(T_array, dC_array_2/C_temp_eg*100, color = "g", label = "8 periods")
plt.xlabel("periods")
plt.ylabel("percent deviation from ss")
plt.legend()
plt.savefig("2e1.pdf")


# In[484]:


dill.dump_session(filepath)


# In[ ]:




