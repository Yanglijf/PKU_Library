# Assignment #D: 十全十美 

Updated 1254 GMT+8 Dec 17, 2024

2024 fall, Complied by <mark>杨李捷飞 工学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 02692: 假币问题

brute force, http://cs101.openjudge.cn/practice/02692

思路：遍历12枚硬币，对每个硬币，假定其重或轻，若假定成立则输出

用时：50分钟

代码：

```python
def check(coin,assume,est1,est2,est3):
    for l, r, result in [est1, est2, est3]:
        l_num = l.count(coin)
        r_num = r.count(coin)
        if l_num == r_num and result != 'even':
            return False
        if assume == 'heavy':
            if l_num > r_num and result != 'down':
                return False
            if l_num < r_num and result != 'up':
                return False
        if assume == 'light':
            if l_num > r_num and result != 'up':
                return False
            if l_num < r_num and result != 'down':
                return False
    return True
n = int(input())
for _ in range(n):
    est1 = input().split()
    est2 = input().split()
    est3 = input().split()
    for coin in 'ABCDEFGHIJKL':
        if check(coin,'heavy',est1,est2,est3):
            print(f'{coin} is the counterfeit coin and it is light.')
            break
        if check(coin, 'light', est1, est2, est3):
            print(f'{coin} is the counterfeit coin and it is heavy.')
            break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241217231549846](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241217231549846.png)



### 01088: 滑雪

dp, dfs similar, http://cs101.openjudge.cn/practice/01088

思路：dfs+记忆化搜索即可

用时：40分钟

代码：

```python
def in_matrix(x, y):
    return 0 <= x < r and 0 <= y < c

def max_lth(x, y):
    if dp[x][y] != 0:
        return dp[x][y]

    max_len = 1
    for dx, dy in move:
        nx, ny = x + dx, y + dy
        if in_matrix(nx, ny) and matrix[nx][ny] < matrix[x][y]:
            max_len = max(max_len, 1 + max_lth(nx, ny))

    dp[x][y] = max_len
    return max_len


r, c = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(r)]
move = [(1, 0), (-1, 0), (0, 1), (0, -1)]
dp = [[0] * c for _ in range(r)]
result = 0
for x in range(r):
    for y in range(c):
        result = max(result, max_lth(x, y))
print(result)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241217232347418](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241217232347418.png)



### 25572: 螃蟹采蘑菇

bfs, dfs, http://cs101.openjudge.cn/practice/25572/

思路：普通的bfs，只是需要一次性处理两个坐标

用时：10分钟

代码：

```python
from collections import deque

move = [(1,0),(-1,0),(0,1),(0,-1)]

def in_matrix(x,y):
    if 0 <= x < n and 0 <= y < n and matrix[x][y] != 1:
        return True
    return False

def is_end(x,y):
    if matrix[x][y] == 9:
        return True
    return False

def can(x1,y1,x2,y2):
    queue = deque([(x1,y1,x2,y2)])
    visited = set()
    while queue:
        x,y,xx,yy = queue.popleft()
        visited.add((x,y,xx,yy))
        for t in range(4):
            dx, dy = move[t]
            nx,ny,nxx,nyy = x + dx,y + dy,xx + dx,yy + dy
            if in_matrix(nx,ny) and in_matrix(nxx,nyy) and (nx,ny,nxx,nyy) not in visited:
                queue.append((nx,ny,nxx,nyy))
                if is_end(nx,ny) or is_end(nxx,nyy):
                    return 'yes'
    return 'no'


n = int(input())
matrix = [list(map(int, input().split())) for _ in range(n)]
start = []
for i in range(n):
    for j in range(n):
        if matrix[i][j] == 5:
            start.append((i,j))
            for t in range(4):
                dx,dy = move[t]
                ni,nj = i + dx,j +dy
                if in_matrix(ni,nj) and matrix[ni][nj] == 5:
                    start.append((ni,nj))
                    break
            break
result = can(start[0][0],start[0][1],start[1][0],start[1][1])
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Downloads\屏幕截图_17-12-2024_232537_cs101.openjudge.cn.jpeg)



### 27373: 最大整数

dp, http://cs101.openjudge.cn/practice/27373/

思路：先按字典序排序，在使用dp考虑位数为k是的最大值

用时：50分钟

代码：

```python
def max_num(m, n, nums):
    nums = list(map(str, nums))
    nums.sort(key=lambda x: x * 20, reverse=True)
    dp = [''] * (m + 1)
    dp[0] = ''
    for num in nums:
        lth = len(num)
        for i in range(m, lth - 1, -1):
            if dp[i - lth] != '' or i - lth == 0:
                dp[i] = max(dp[i], dp[i - lth] + num)
    return max(dp[1:], key=lambda x: (len(x), x))

m = int(input())
n = int(input())
nums = list(map(int, input().split()))
result = max_num(m, n, nums)
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241217233849555](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241217233849555.png)



### 02811: 熄灯问题

brute force, http://cs101.openjudge.cn/practice/02811

思路：后续情形由第一行的按法确定，因此遍历第一行即可

用时：30分钟

代码：

```python
def in_matrix(x,y):
    if 0 <= x < 5 and 0 <= y < 6:
        return True
    return False

def change(x,y):
    changes = [(x,y)]
    for dx,dy in move:
        nx,ny = x+dx,y+dy
        if in_matrix(nx,ny):
            changes.append((nx,ny))
    return changes

def press(x,y,current):
    for nx,ny in change(x,y):
        current[nx][ny] = 1 - current[nx][ny]

def check(current):
    for l in current:
        if any(l):
            return False
    return True

def press_plan():
    for i in range(1<<6):
        presses = [[0] * 6 for _ in range(5)]
        ini_matrix = [l[:] for l in matrix]
        for c in range(6):
            if (i>>c) & 1:
                presses[0][c] = 1
                press(0,c,ini_matrix)
        for k in range(1, 5):
            for t in range(6):
                if ini_matrix[k-1][t] == 1:
                    presses[k][t] = 1
                    press(k,t,ini_matrix)
        if check(ini_matrix):
            return presses
    return None

move = [(0,1),(0,-1),(1,0),(-1,0)]
matrix = [list(map(int, input().split())) for _ in range(5)]
plan = press_plan()
for l in plan:
    print(*l)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Downloads\屏幕截图_17-12-2024_234251_cs101.openjudge.cn.jpeg)



### 08210: 河中跳房子

binary search, greedy, http://cs101.openjudge.cn/practice/08210/

思路：二分查找，判断是否可以达到某个距离

用时：30分钟

代码：

```python
def check(d):
    count = 0
    current = 0
    for i in range(1,n+2):
        if stones[i] - current < d:
            count += 1
        else:
            current = stones[i]
    if count > m:
        return False
    return True

l, n, m = map(int, input().split())
stones = [0] + [int(input()) for _ in range(n)] + [l]
left, right = 1,l
result = 1
while left <= right:
    mid = (left+right)//2
    if check(mid):
        result = mid
        left = mid + 1
    else:
        right = mid - 1
print(result)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241217234501518](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241217234501518.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

这段时间每日选做能及时跟进，也选做了往年的考试题目

下周就要机考，感觉准备还是不是很充分



