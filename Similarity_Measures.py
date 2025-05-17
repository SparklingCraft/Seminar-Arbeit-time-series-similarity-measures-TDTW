import numpy as np

class Time_Series_Similarity:
    def __init__(self, distance_matrix: np.array, Cost_matrix: np.array, allignment_A, allignment_B, path, distance: float,name: str ):
        self.distance_matrix = distance_matrix
        self.Cost_matrix = Cost_matrix
        self.allignment_A = allignment_A
        self.allignment_B = allignment_B
        self.path = path
        self.distance = distance
        self.name = name

    def __str__(self):
        return f"Distance Matrix: {self.distance_matrix}\nCost Matrix: {self.Cost_matrix}\nAlignment A: {self.allignment_A}\nAlignment B: {self.allignment_B}\nPath: {self.path}\nDistance: {self.distance}"
    # getters
    def get_distance_matrix(self):
        return self.distance_matrix
    def get_Cost_matrix(self):
        return self.Cost_matrix
    def get_allignment_A(self):
        return self.allignment_A
    def get_allignment_B(self):
        return self.allignment_B
    def get_path(self):
        return self.path
    def get_distance(self):
        return self.distance
    def get_name(self):
        return self.name
    def len(self):
        return len(self.path)
    
def DTW(A: np.array, B: np.array) -> Time_Series_Similarity:
    """
    Computes the Dynamic Time Warping (DTW) distance between two time series.

    Parameters:
    A (np.array): First time series.
    B (np.array): Second time series.

    Returns:
    Time_Series_Similarity: Object containing the DTW distance and alignment information.
    """
    
    n = len(A)
    m = len(B)
    D=np.zeros((n, m))
    R=np.zeros((n, m))
   
    D[0, 0] = (A[0]-B[0])**2
    D[0][0]=(A[0]-B[0])**2

    R[0][0]=D[0][0]
    #pointer        
    min_pointer_matrix=np.zeros((n,m))
    for i in range(1, n):
        D[i, 0] = (A[i]-B[0])**2
        R[i][0]=R[i-1][0]+D[i][0]
        # References for the allignment path
        min_pointer_matrix[i][0]=2
    for j in range(1, m):
        D[0, j] = (A[0]-B[j])**2
        R[0][j]=R[0][j-1]+D[0][j]
        # References for the allignment path
        min_pointer_matrix[0][j]=1
    for i in range(1,n):
        for j in range(1,m):
            D[i][j]=(A[i]-B[j])**2
            R[i][j]=D[i][j]+min(R[i-1][j-1],R[i][j-1],R[i-1][j])
            # References for the allignment path
            min_pointer_matrix[i][j]=np.argmin([R[i-1][j-1],R[i][j-1],R[i-1][j]])
    

    # reconstuction of the path
    i=n-1
    j=m-1
    allignment_A=[]
    allignment_B=[]
    path=[]
    while i>0 or j>0:
        allignment_A.append(i)
        allignment_B.append(j)
        path.append([i,j])
        if i==0:
            j=j-1
        elif j==0:
            i=i-1
        else:
            if min_pointer_matrix[i][j]==0:
                i=i-1
                j=j-1
            elif min_pointer_matrix[i][j]==1:
                j=j-1
            else:
                i=i-1
    allignment_A.append(i)
    allignment_B.append(j)
    path.append([i,j])
    return Time_Series_Similarity(D, R, allignment_A, allignment_B, path, R[n-1][m-1],"DTW")

def TDTW(A: np.array, B: np.array) -> Time_Series_Similarity:
    """
    Computes the Dynamic Time Warping (DTW) distance between two time series.

    Parameters:
    A (np.array): First time series.
    B (np.array): Second time series.

    Returns:
    Time_Series_Similarity: Object containing the DTW distance and alignment information.
    """
    n = len(A) 
    m = len(B)
    A_r=A.copy()[::-1]
    B_r=B.copy()[::-1]
   
    D=np.zeros((n,m))
    R=np.zeros((n,m))
    W=np.zeros((n,m))
    WxD=np.zeros((n,m))
    parent_matrix=np.zeros((n,m,2))

    R[0,:]=np.inf
    R[:,0]=np.inf

    # matrx calculation
    for i in range(n):
        x=(1-(i/n))**2
        for j in range(m):
            y=(1-(j/m))**2
            W[i][j]=(np.sqrt(x+y))/np.sqrt(2)

            D[i][j]=(A_r[i]-B_r[j])**2

            WxD[i][j]= W[i][j]* D[i][j]
            R[i][j]= WxD[i][j]+min(R[i-1][j-1],R[i][j-1],R[i-1][j])
            # References for the allignment path
            if np.argmin([R[i-1][j-1],R[i][j-1],R[i-1][j]])==0:
                parent_matrix[i][j][0]=i-1
                parent_matrix[i][j][1]=j-1
            elif np.argmin([R[i-1][j-1],R[i][j-1],R[i-1][j]])==1:
                parent_matrix[i][j][0]=i
                parent_matrix[i][j][1]=j-1
            else:
                parent_matrix[i][j][0]=i-1
                parent_matrix[i][j][1]=j
    # reconstuction of the path
    i=n-1
    j=m-1
    allignment_A=[]
    allignment_B=[]
    path=[]
    while i>0 or j>0:
        allignment_A.append(n-i-1)
        allignment_B.append(m-j-1)
        path.append([n-i-1,m-j-1])
        if i==0:
            j=j-1
        elif j==0:
            i=i-1
        else:
            if parent_matrix[i][j][0]==i-1 and parent_matrix[i][j][1]==j-1:
                i=i-1
                j=j-1
            elif parent_matrix[i][j][0]==i and parent_matrix[i][j][1]==j-1:
                j=j-1
            else:
                i=i-1
    allignment_A.append(n-i-1)
    allignment_B.append(m-j-1)
    path.append([n-i-1,m-j-1])
    return Time_Series_Similarity(WxD, R, allignment_A, allignment_B, path, R[n-1][m-1],"TDTW")

def WDTW(A: np.ndarray, B: np.ndarray, g: float) -> Time_Series_Similarity:
    
    def dist(x,y):
        return (A[x]-B[y])**2
    x_size = len(A)
    y_size = len(B)
    D = np.zeros((x_size+1, y_size+1))
    R = np.full((x_size + 1, y_size + 1), np.inf)
    R[0, 0] = 0.0

    max_size = max(x_size, y_size)
    W = np.array(
        [1 / (1 + np.exp(-g * (i - max_size / 2))) for i in range(0, max_size)]
    )

    for i in range(x_size):
        for j in range(y_size):        
            D[i+1, j+1] = dist(A[i], B[j]) * W[abs(i - j)]
            R[i + 1, j + 1] = dist(
                A[i], B[j]
            ) * W[abs(i - j)] + min(
                R[i, j + 1],
                R[i + 1, j],
                R[i, j],
            )
    n=x_size
    m=y_size
    i = n - 1
    j = m - 1
    path=[]
    allignment_A=[]
    allignment_B=[]
    size_warping_path = 1
    # remove the first row and first collum from the distance matrix

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(R[i-1,j-1], R[i-1,j], R[i,j-1])
            if R[i-1,j-1] == min_val:
                i -= 1
                j -= 1
            elif R[i,j-1] == min_val:
                j -= 1
            else:
                i -= 1
        path.append([i,j])
        allignment_A.append(i)
        allignment_B.append(j)
        size_warping_path += 1
    return Time_Series_Similarity(D, R, allignment_A, allignment_B, path, R[-1][-1],"WDTW")

def Eucledian(A: np.array, B: np.array) -> Time_Series_Similarity:
    """
    Computes the Euclidean distance between two points.

    Parameters:
    A (float): First time series.
    B (float): Second time series.

    Returns:
    Time_Series_Similarity: Object containing the DTW distance and alignment information.
    """
    n = len(A)
    m = len(B)
    def dist(x,y):
        return ((A(x)-B(y))**2)
    
    if n != m:
        raise ValueError("Input arrays must have the same length.")
    else:
        dist_val=0
        allignment_A=[]
        allignment_B=[]
        path=[]
        for i in range(n):
            dist_val+=dist(A[i],B[i])
            
            allignment_A.append(i)
            allignment_B.append(i)
            path.append([i,i])
        distance_matrix = np.zeros((n, m))
        for i in range(0,n):
            for j in range(0,m):
                distance_matrix[i][j]=dist(A[i],B[j])
    return Time_Series_Similarity(distance_matrix, distance_matrix, allignment_A, allignment_B, path, dist_val,"Euclidean") 

def LDTW(A: np.array, B: np.array,max_L=-1) -> Time_Series_Similarity:
    """
    Computes the Dynamic Time Warping (DTW) distance between two time series.

    Parameters:
    A (np.array): First time series.
    B (np.array): Second time series.
    max_L (int): Maximum length of the warping path.

    Returns:
    Time_Series_Similarity: Object containing the DTW distance and alignment information.
    """
    # shift the indexes of both Ts to the right by 1
    A = np.insert(A, 0, 0)
    B = np.insert(B, 0, 0)

    if max_L==-1:
        max_L=int(max(len(A),len(B))*1.5)
    
    def dist(x,y):
        dist=(A[x]-B[y])**2
        return dist
       

    n = len(A)
    m = len(B)
    D=np.zeros((n, m))
    R=np.zeros((n, m,max_L))
    
    D[1, 1] = dist(A[1],B[1])
    # m[1][1][0]=distance_matrix[0][0]
    for i in range(0,n):
        for j in range(0,m):
            D[i][j]=dist(A[i],B[j])
            for s in range(0,max_L):
                R[i][j][s]=np.inf
    R[1][1][0]=D[1][1]
    
    #pointer        
    min_pointer_matrix=np.zeros((n,m,max_L))

    def min_Steps(i,j):
        return max(i,j) - 1 # dummy function for now
    def max_steps_No_Limit(i,j):
        if i== 0 or j == 0:
            return max(i,j)
        else:
            return i + j - 1
        

    def max_Steps(i,j):
        candiate1=max_steps_No_Limit(i,j)
        candiate2=max_L-1 -max(n-i,m-j)
        return min(candiate1,candiate2)

    # Initialize first row and column
    for i in range(2, n):
        D[i, 1] = dist(A[i],B[0])
        R[i][1][i-1]=R[i-1][1][i-2]+D[i][1]
        min_pointer_matrix[i][1][i-1]=2
    for j in range(2, m):
        D[1, j] = dist(A[0],B[j])
        R[1][j][j-1]=R[1][j-1][j-2]+D[1][j]
        min_pointer_matrix[1][j][j-1]=1

    # Calculate DTW matrix with constraints
    for i in range(2,n):
        for j in range(2,m):
            min_s=min_Steps(i,j)
            max_s=max_Steps(i,j)
            distance_loc=dist(A[i],B[j])
            D[i][j]=distance_loc

            for s in range(min_s,max_s):
                R[i][j][s]=D[i][j]+min(R[i-1][j-1][s-1],R[i][j-1][s-1],R[i-1][j][s-1])
                min_pointer_matrix[i][j][s]=np.argmin([R[i-1][j-1][s-1],R[i][j-1][s-1],R[i-1][j][s-1]])       

    min_s_final=max(n,m)
    max_s_final=max_L-1
    ldtw_disrance=float("inf")
    path_length=0
    for s in range(min_s_final,max_s_final+1):
        if R[n-1][m-1][s]<ldtw_disrance:
            ldtw_disrance=R[n-1][m-1][s]
            path_length=s  
     

    i=n-1
    j=m-1
    reverse_s =path_length+1
    allignment_A=[]
    allignment_B=[]
    path=[]
    while i>0 or j>0:
        reverse_s=reverse_s-1
        allignment_A.append(i-1)
        allignment_B.append(j-1)
        path.append([i-1,j-1])
        if i==0:
            j=j-1
        elif j==0:
            i=i-1
        else:
            if min_pointer_matrix[i][j][reverse_s]==0:
                i=i-1
                j=j-1
            elif min_pointer_matrix[i][j][reverse_s]==1:
                j=j-1
            else:
                i=i-1
    cost_matrix = np.zeros((n-1, m-1))
    for i in range(1,n):
        for j in range(1,m):
            cost_matrix[i-1][j-1]=min(R[i][j])
            
    return Time_Series_Similarity(D, cost_matrix, allignment_A, allignment_B, path, ldtw_disrance,"LDWT")

def DDDTW(A, B,dddtw_apha) -> Time_Series_Similarity:
    """
    Computes the Dynamic Time Warping (DTW) distance between two time series using derivative distance.

    Parameters:
    ts_A (np.array): First time series.
    ts_B (np.array): Second time series.
    dddtw_apha (float): Weighting factor for the derivative distance.

    Returns:
    Time_Series_Similarity: Object containing the DTW distance and alignment information.
    """
    def dist(x,y):
        return (A[x]-B[y])**2
    n = len(A)
    m = len(B)
    distance_matrix=np.zeros((n, m))
    derivative_distance_matrix=np.zeros((n, m))
    normal_distance_matrix=np.zeros((n, m))
    sum_distance_matrix=np.zeros((n, m))
   
    normal_distance_matrix[0][0]=dist(A[0],B[0])
    sum_distance_matrix[0][0]=normal_distance_matrix[0][0]
    #pointer        
    min_pointer_matrix=np.zeros((n,m))

    # Calculate derivatives
    for i in range(n):
        if i>0 and i<n-1:
            d_a_i=(A[i]-A[i-1])+(A[i+1]-A[i])/2
        elif i==0:
            d_a_i=(A[i+1]-A[i])
        elif i==n-1:
            d_a_i=(A[i]-A[i-1])
        for j in range(m):
            if j>0 and j<m-1:
                d_b_j=(B[j]-B[j-1])+(B[j+1]-B[j])/2
            elif j==0:
                d_b_j=(B[j+1]-B[j])
            elif j==m-1:
                d_b_j=(B[j]-B[j-1])
            derivative_distance_matrix[i,j]=(d_a_i-d_b_j)**2
            normal_distance_matrix[i][j]=dist(A[i],B[j])


    for i in range(1, n):
        distance_matrix[i, 0] =(1-dddtw_apha) *normal_distance_matrix[i][0]+dddtw_apha*derivative_distance_matrix[i][0]
        sum_distance_matrix[i][0]=sum_distance_matrix[i-1][0]+distance_matrix[i][0]
        min_pointer_matrix[i][0]=2
    for j in range(1, m):
        distance_matrix[0, j] = (1-dddtw_apha) *normal_distance_matrix[0][j]+dddtw_apha*derivative_distance_matrix[0][j]
        sum_distance_matrix[0][j]=sum_distance_matrix[0][j-1]+distance_matrix[0][j]
        min_pointer_matrix[0][j]=1
    for i in range(1,n):
        for j in range(1,m):
            distance_matrix[i][j]= (1-dddtw_apha) *normal_distance_matrix[i][j]+dddtw_apha*derivative_distance_matrix[i][j]
            sum_distance_matrix[i][j]=distance_matrix[i][j]+min(sum_distance_matrix[i-1][j-1],sum_distance_matrix[i][j-1],sum_distance_matrix[i-1][j])
            min_pointer_matrix[i][j]=np.argmin([sum_distance_matrix[i-1][j-1],sum_distance_matrix[i][j-1],sum_distance_matrix[i-1][j]])
    i=n-1
    j=m-1
    allignment_A=[]
    allignment_B=[]
    path=[]
    while i>0 or j>0:
        allignment_A.append(i)
        allignment_B.append(j)
        path.append([i,j])
        if i==0:
            j=j-1
        elif j==0:
            i=i-1
        else:
            if min_pointer_matrix[i][j]==0:
                i=i-1
                j=j-1
            elif min_pointer_matrix[i][j]==1:
                j=j-1
            else:
                i=i-1
    return Time_Series_Similarity(distance_matrix, sum_distance_matrix, allignment_A, allignment_B, path, sum_distance_matrix[n-1][m-1],"DDDTW")

