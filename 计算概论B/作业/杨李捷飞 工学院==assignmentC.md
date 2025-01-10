# Assignment #C: 五味杂陈 

Updated 1148 GMT+8 Dec 10, 2024

2024 fall, Complied by <mark>同学的姓名、院系</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 1115. 取石子游戏

dfs, https://www.acwing.com/problem/content/description/1117/

思路：当大数被小数整除，先手胜；当大数大于小数的两倍，先手有两种选择，先手胜。然后划归到较小情形

用时：20分钟

代码：

```python
def win(a,b):
    a,b = max(a,b),min(a,b)
    if b == 0 or a//b >= 2 or a == b:
        return True
    a,b = b,a%b
    return not win(a,b)
while True:
    a, b = map(int ,input().split())
    if a==0 and b == 0:
        break
    if win(a,b):
        print('win')
    else:
        print('lose')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\Screenshot 2024-12-10 152443.png)



### 25570: 洋葱

Matrices, http://cs101.openjudge.cn/practice/25570

思路：算出层数并将每一层的和初始化为0，遍历矩阵，通过运算行列坐标得出其属于的层并相加，最后输出最大值

用时：25分钟

代码：

```python
n = int(input())
matrix = [list(map(int, input().split())) + [-1] for _ in range(n)]
t = (n+1)//2
result = [0]*(t)
for i in range(n):
    for j in range(n):
        k = min(i,j,n-i-1,n-j-1)
        result[k] += matrix[i][j]
print(max(result))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241210152959995](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241210152959995.png)



### 1526C1. Potions(Easy Version)

greedy, dp, data structures, brute force, *1500, https://codeforces.com/problemset/problem/1526/C1

思路：贪心算法，若喝下当前药水生命为负，则放弃喝下的最小数值的药水

用时：20分钟

代码：

```python
import heapq
n = int(input())
a = list(map(int, input().split()))
health = 0
count = 0
potions = []
for i in range(n):
    health += a[i]
    if health >= 0:
        count += 1
        heapq.heappush(potions,a[i])
    else:
        min_potion = heapq.heappushpop(potions,a[i])
        health -= min_potion
print(count)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241210153335278](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241210153335278.png)



### 22067: 快速堆猪

辅助栈，http://cs101.openjudge.cn/practice/22067/

思路：构造两个栈，更新堆的猪的重量和当前最小重量，pop时同时pop即可

用时：20分钟

代码：

```python
stack = []
stack_min = []
try:
    while True:
        s = input().split()
        if s[0] == 'pop':
            if stack:
                stack.pop()
                stack_min.pop()
        elif s[0] == 'min':
            if stack:
                print(stack_min[-1])
        else:
            weight = int(s[1])
            stack.append(weight)
            if stack_min:
                stack_min.append(min(stack_min[-1],weight))
            else:
                stack_min.append(weight)
except EOFError:
    pass
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241210174549480](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241210174549480.png)



### 20106: 走山路

Dijkstra, http://cs101.openjudge.cn/practice/20106/

思路：每次扩展的是当前路径消耗最小的节点，加入剪枝

用时：40分钟

代码：

```python
import heapq

movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
def dijkstra(x1, y1, x2, y2):
    if matrix[x1][y1] == '#' or matrix[x2][y2] == '#':
        return 'NO'
    if (x1,y1) == (x2,y2):
        return 0
    pq = [(0,x1,y1)]
    visited = set()
    min_power = [[float('inf')] * n for _ in range(m)]
    min_power[x1][y1] = 0
    while pq:
        power,x,y = heapq.heappop(pq)
        if (x,y) in visited:
            continue
        visited.add((x,y))
        if (x,y) == (x2,y2):
            return power

        for dx,dy in movements:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] != '#' and (nx, ny) not in visited:
                npower = power + abs(int(matrix[nx][ny])-int(matrix[x][y]))
                if npower < min_power[nx][ny]:
                    min_power[nx][ny] = npower
                    heapq.heappush(pq,(npower,nx,ny))
    return 'NO'



m, n, p = map(int, input().split())
matrix = [list(input().split()) for _ in range(m)]
for _ in range(p):
    x1,y1, x2,y2 = map(int,input().split())
    result = dijkstra(x1,y1,x2,y2)
    print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241210175552920](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241210175552920.png)



### 04129: 变换的迷宫

bfs, http://cs101.openjudge.cn/practice/04129/

思路：找到起点，将时间%4和坐标一起放入visited进行bfs

用时：35分钟

代码：

```python
from functools import lru_cache
from collections import deque

movements = [(1,0),(-1,0),(0,1),(0,-1)]

def min_time(x1,y1):
    queue = deque([(0,x1,y1)])
    visited = {(0,x1,y1)}
    while queue:
        time,x,y = queue.popleft()
        for i in range(4):
            nx,ny = x + movements[i][0],y+movements[i][1]
            temp = (time + 1)%k
            if 0 <= nx < r and 0 <= ny < c and (temp, nx, ny) not in visited:
                if matrix[nx][ny] == 'E':
                    return time+1
                elif matrix[nx][ny] != '#' or temp == 0:
                    queue.append((time + 1,nx,ny))
                    visited.add((temp,nx,ny))
    return 'Oop!'

t = int(input())
for _ in range(t):
    r,c,k = map(int ,input().split())
    matrix = []
    for i in range(r):
        s = list(input())
        matrix.append(s)
        if 'S' in s:
            x1 = i
            y1 = s.index('S')
    print(min_time(x1,y1))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241210174953646](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241210174953646.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

这次作业思路还是比较好想，但是代码还是维护了很多次才ac

加练了一些leetcode和cf上的题目，感觉有一些提升



