# Assignment #9: dfs, bfs, & dp

Updated 2107 GMT+8 Nov 19, 2024

2024 fall, Complied by <mark>同学的姓名、院系</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 18160: 最大连通域面积

dfs similar, http://cs101.openjudge.cn/practice/18160

思路：对每一个点进行深度优先搜索，计算其联通面积，最后取极值

用时：30分钟左右

代码：

```python
import sys
sys.setrecursionlimit(10000)
movements = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
result = 0
def dfs(lakes,x,y,n,m):
    global result
    if not (0 <= x < n and 0 <= y < m) or lakes[x][y] != 'W':
        return
    lakes[x][y] = '.'
    result += 1
    for m_x, m_y in movements:
        new_x, new_y = x + m_x, y + m_y
        if 0 <= new_x < n and 0 <= new_y < m and lakes[new_x][new_y] == 'W':
            dfs(lakes,new_x,new_y,n,m)
t = int(input())
for _ in range(t):
    n, m = map(int, input().split())
    ans = 0
    lakes = []
    for _ in range(n):
        s = list(input().strip())
        lakes.append(s)
    for i in range(n):
        for j in range(m):
            if lakes[i][j] == 'W':
                result = 0
                dfs(lakes, i, j, n, m)
                ans = max(ans,result)
    print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241119230343678](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241119230343678.png)



### 19930: 寻宝

bfs, http://cs101.openjudge.cn/practice/19930

思路：从左上角开始进行bfs,记录步骤

用时：35分钟

代码：

```python
movements = [[-1,0],[1,0],[0,1],[0,-1]]
m, n = map(int, input().split())
matrix = [list(map(int, input().split())) + [2] for i in range(m)]
queue = [(0,0,0)]
inq = [[False for _ in range(n)] for _ in range(m)]
inq[0][0] = True
while queue:
    x,y,steps = queue.pop(0)
    if matrix[x][y] == 1:
        print(steps)
        break
    for dx, dy in movements:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < m and 0 <= new_y < n and matrix[new_x][new_y] != 2 and not inq[new_x][new_y]:
            queue.append((new_x, new_y, steps + 1))
            inq[new_x][new_y] = True
else:
    print('NO')
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241119234112266](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241119234112266.png)



### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123

思路：定义行走方式，从给定坐标进行深度优先搜索，搜索次数达到总坐标数即为一条路径

用时：20分钟

代码：

```python
movements = [[1,2],[1,-2],[2,1],[2,-1],[-1,2],[-1,-2],[-2,1],[-2,-1]]
result = 0
def dfs(x,y,step):
    global result
    if step == n*m:
        result += 1
        return
    for dx,dy in movements:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < n and 0 <= new_y < m and matrix[new_x][new_y] == 0:
            matrix[new_x][new_y] = 1
            dfs(new_x,new_y,step+1)
            matrix[new_x][new_y] = 0
t = int(input())
for _ in range(t):
    n, m, x, y = map(int, input().split())
    matrix = [[0]*m for _ in range(n)]
    result = 0
    matrix[x][y] = 1
    dfs(x,y,1)
    print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241120001034564](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241120001034564.png)



### sy316: 矩阵最大权值路径

dfs, https://sunnywhy.com/sfbj/8/1/316

思路：构造二维visited对应急矩阵大小，使用dfs并修改路径和最大权重

用时：45分钟

代码：

```python
n, m = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(n)]
visited = [[False for _ in range(m)] for _ in range(n)]
max_weight = float('-inf')
max_path = []
def dfs(x,y,current_weight,current_path):

    global max_weight, max_path

    if x < 0 or x >= n or y < 0 or y >= m or visited[x][y]:
        return

    if x == n-1 and y == m-1:
        current_weight += matrix[x][y]
        if current_weight > max_weight:
            max_weight = current_weight
            max_path = current_path + [(x+1, y+1)]
        return

    visited[x][y] = True
    current_path.append((x+1,y+1))

    dfs(x - 1, y, current_weight + matrix[x][y], current_path)
    dfs(x + 1, y, current_weight + matrix[x][y], current_path)
    dfs(x, y - 1, current_weight + matrix[x][y], current_path)
    dfs(x, y + 1, current_weight + matrix[x][y], current_path)

    visited[x][y] = False
    current_path.pop()


dfs(0,0,0,[])
for x, y in max_path:
    print(x,y)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241120162155791](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241120162155791.png)





### LeetCode62.不同路径

dp, https://leetcode.cn/problems/unique-paths/

思路：组合数直接计算

用时：5分钟

代码：

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        import math   
        result = math.comb(m+n-2,m-1)
        return result   
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241120001526885](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241120001526885.png)



### sy358: 受到祝福的平方

dfs, dp, https://sunnywhy.com/sfbj/8/3/539

思路：使用dp来判断当前长度是否可以被分割

用时：15分钟

代码：

```python
import math
def judge(x):
    if x == 0:
        return False
    rt = int(math.sqrt(x))
    return rt*rt == x
A = int(input())
s = str(A)
lth = len(s)
dp = [False]*(lth+1)
dp[0] = True
for i in range(1,lth+1):
    for j in range(i):
        ss = s[j:i]
        if dp[j] and judge(int(ss)):
            dp[i] = True
            break
if dp[lth]:
    print('Yes')
else:
    print('N0')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241120164213059](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241120164213059.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

这周每日选做可以按时跟进，加练cf上的dfs的相关题目对做题帮助很大

代码书写时细节方面处理不够好，经常出现无法处理特殊情况的错误



