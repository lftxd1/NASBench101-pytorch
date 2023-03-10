import random


INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
CONV5X5= 'conv3x3-bn-relu'

is_find=False
def isValid(matrix):
    """判断邻接表的有效性"""
    global is_find
    is_find=False
    edge=0
    for i in range(7):
        for j in range(i+1,7):
            if matrix[i][j]==1:
                edge+=1
    if edge>9 or edge<3:
        return False
    
    """使用深度优先搜索输入节点是否连接到输出节点"""
    def dfs(i):
        if i==len(matrix)-1:
            global is_find
            is_find=True
            return
        
        for j in range(len(matrix)):
            if matrix[i][j]==1:
                dfs(j)
    dfs(0)
    return is_find

def GeneratePerson():
    matrix=[[0]*7 for i in range(7)]
    ops=[CONV1X1, CONV3X3,  MAXPOOL3X3]
    while isValid(matrix)==False:
        for i in range(7):
            for j in range(i+1,7):
                matrix[i][j]=random.choice([0,1])
    res_ops=[]
    for i in range(5):
        op_name=random.choice(ops)
        if op_name in res_ops:
            ans=0
            for i in res_ops:
                if i.find(op_name)!=-1:
                    ans+=1
            op_name=op_name+"#"+str(ans)
        res_ops.append(op_name)
        
    res_ops.insert(0,INPUT)
    res_ops.append(OUTPUT)
    return matrix,res_ops

