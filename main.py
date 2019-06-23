import numpy as np

def step_gradient(data,learning_rate,m):
    rows = len(data)
    col = len(data[0])
    m_slope = np.zeros(col-1)
    
    for i in range(rows):
        prod_sum = 0
        x = data[i,:col-1]
        y = data[i,col-1]
        prod_sum = (m*x).sum()
        for j in range(col-1):
            m_slope[j] += (-2/rows)*(y - prod_sum)*x[j]
    m_new = []
    m_new = m
    for i in range(col-1):
        m_new[i] = m[i] - learning_rate*m_slope[i]
        
    return m_new  
    
def cost(data,m):
    total_cost = 0
    rows = len(data)
    col = len(data[0])
    
    for i in range(rows):
        s_sum = 0
        x = data[i,:col-1]
        y = data[i,col-1]
        s_sum = (m*x).sum()
        total_cost += (1/rows)*((y - s_sum)**2)
        
    return total_cost 

def gd(data,learning_rate,num_iterations):
    col = len(data[0])
    m = np.zeros(col-1)
    
    for i in range(num_iterations):
        m = step_gradient(data,learning_rate,m)
        print(cost(data,m))
        
    return m    		
    
def run():    
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    
    data = np.loadtxt("ccpp_train.csv",delimiter = ",")
    rows = len(data)
    col = len(data[0])
    #print(rows,col)
    ones = np.ones((rows,col+1))
    ones[:,:col-1] = data[:,:col-1]
    ones[:,col:col+1] = data[:,col-1:]
    #rows_ones = len(ones)
    #col_ones = len(ones[0])
    #print(rows_ones,col_ones)
    #print(ones)
    #scaler.fit(ones)
    #scaler.transform(ones)
    learning_rate = 0.00000095699
    num_iterations = 100000
    
    m = gd(ones,learning_rate,num_iterations)
    return m
ans = run()
def predict(x_array):
    m = run()
    rows = len(x_array)
    col = len(x_array[0])
    y = np.zeros(rows)
    for i in range(rows):
        x = x_array[i,:]
        y[i] += (m*x).sum()
    
    return y
		
data2 = np.loadtxt("ccpp_test.csv",delimiter = ",")
rows = len(data2)
col = len(data2[0])
ones = np.ones((rows,col+1))
ones[:,:col] = data2[:,:col]
result = predict(ones)
np.savetxt("result.csv",result,fmt="%.1f", delimiter=',', newline='\n')   
        
