# Assignment #10: dp & bfs

Updated 2 GMT+8 Nov 25, 2024

2024 fall, Complied by <mark>同学的姓名、院系</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### LuoguP1255 数楼梯

dp, bfs, https://www.luogu.com.cn/problem/P1255

思路：找出递推公式即可

用时：3分钟

代码：

```python
dp = [0]*(5000+1)
dp[1] = 1
dp[2] = 2
for i in range(3,5001):
    dp[i] = dp[i-1] + dp[i-2]
n = int(input())
print(dp[n])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-11-26 125334.png)



### 27528: 跳台阶

dp, http://cs101.openjudge.cn/practice/27528/

思路：注意到n节台阶又2的n-1次方种方式即可

用时：3分钟

代码：

```python
n = int(input())
print(2**(n-1))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241126130422946](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241126130422946.png)



### 474D. Flowers

dp, https://codeforces.com/problemset/problem/474/D

思路：构造dp数组，对i，若前一个为r，则有i-1时的数量；若为w，则为i-k时种数。利用前缀和

用时：30分钟

代码：

```python
mod = 1000000007
t, k = map(int, input().split())
dp = [0]*(100001)
dp[0] = 1
pre_list = [0]*(100001)
for i in range(1,100001):
    dp[i] = dp[i-1]
    if i >= k:
        dp[i] = (dp[i] + dp[i-k])%mod
    pre_list[i] = pre_list[i-1] + dp[i]
for _ in range(t):
    a, b = map(int, input().split())
    result = (pre_list[b] - pre_list[a-1])%mod
    print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241126131300953](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241126131300953.png)



### LeetCode5.最长回文子串

dp, two pointers, string, https://leetcode.cn/problems/longest-palindromic-substring/

思路：二维dp数组判断从i到j是否为回文子串，并更新最大长度和对应字串

用时：30分钟

代码：

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        lth = len(s)
        if lth == 1:
            return s
        max_length = 1
        start = 0
        dp = [[False] * lth for _ in range(lth)]
        for i in range(lth):
            dp[i][i] = True
        for l in range(2, lth + 1):
            for i in range(lth - l + 1):
                j = i + l - 1
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                if dp[i][j] and l > max_length:
                    max_length = l
                    start = i
        return s[start:start + max_length]
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241126151309471](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241126151309471.png)





### 12029: 水淹七军

bfs, dfs, http://cs101.openjudge.cn/practice/12029/

思路：思路就是从出发点沿水的方向更新各点水位，

​            但是没发现需要一次性读取re多次，浪费很长时间

用时：1小时

代码：

```python
import sys
sys.setrecursionlimit(1000000000)

def is_valid(x, y, m, n):
    return 0 <= x < m and 0 <= y < n
def dfs(x, y, water_height_value, m, n, matrix, water_height):
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for i in range(4):
        nx, ny = x + dx[i], y + dy[i]
        if is_valid(nx, ny, m, n) and matrix[nx][ny] < water_height_value:
            if water_height[nx][ny] < water_height_value:
                water_height[x][y] = water_height_value
                dfs(nx, ny, water_height_value, m, n, matrix, water_height)

data = sys.stdin.read().split()
idx = 0
k = int(data[idx])
idx += 1
results = []
for _ in range(k):
    m, n = map(int, data[idx:idx+2])
    idx += 2
    matrix = []
    for i in range(m):
        matrix.append(list(map(int, data[idx:idx + n])))
        idx += n
    water_height = [[0] * n for _ in range(m)]
    i, j = map(int, data[idx:idx + 2])
    idx += 2
    i, j = i - 1, j - 1
    p = int(data[idx])
    idx += 1
    for _ in range(p):
        x, y = map(int, data[idx:idx + 2])
        idx += 2
        x, y = x - 1, y - 1
        if matrix[x][y] <= matrix[i][j]:
            continue
        dfs(x, y, matrix[x][y], m, n, matrix, water_height)
    results.append("Yes" if water_height[i][j] > 0 else "No")
print("\n".join(results))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241126204640591](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241126204640591.png)

![](C:\Users\yangljf\Downloads\屏幕截图_26-11-2024_233729_cs101.openjudge.cn.jpeg)

​     （ps：第一次对是看看题解对不对）

### 02802: 小游戏

bfs, http://cs101.openjudge.cn/practice/02802/

思路：定义四个方向并标号，然后利用bfs进行操作，访问过的点要和来时的方向一起储存，注意要加保护圈

用时：2小时

代码：

```python
from collections import deque
movements = [(0,-1,0),(1,1,0),(2,0,1),(3,0,-1)]
def bfs(matrix,start,end):
    w = len(matrix[0])
    h = len(matrix)
    queue = deque([(start[0], start[1], -1, 0)])
    visited = set()
    ans = []
    while queue:
        x, y, last_direct, counts = queue.popleft()
        if (x,y) == end:
            ans.append(counts)
            break
        for d,dx,dy in movements:
            new_x,new_y = x + dx, y + dy
            if 0 <= new_x < h and 0 <= new_y < w and (((new_x,new_y),d) not in visited):
                if d == last_direct:
                    new_counts = counts
                else:
                    new_counts = counts + 1
                if (new_x,new_y) == end:
                    ans.append(new_counts)
                    continue
                if matrix[new_x][new_y] != 'X':
                    visited.add(((new_x,new_y),d))
                    queue.append((new_x,new_y,d,new_counts))
    if len(ans) == 0:
        return -1
    else:
        return min(ans)

def solve(matrix,pairs):
    results = []
    case = 1
    for y1,x1,y2,x2 in pairs:
        counts = bfs(matrix,(x1, y1),(x2,y2))
        if counts != -1:
            results.append(f'Pair {case}: {counts} segments.')
        else:
            results.append(f'Pair {case}: impossible.')
        case += 1
    return results

board = 1
while True:
    w, h = map(int, input().split())
    if w == h == 0:
        break
    matrix = [' ' * (w + 2)] + [' ' + input() + ' ' for _ in range(h)] + [' ' * (w + 2)]
    pairs = []
    while True:
        pair = list(map(int, input().split()))
        if pair == [0,0,0,0]:
            break
        pairs.append(pair)
    print(f'Board #{board}:')
    results = solve(matrix,pairs)
    for result in results:
        print(result)
    print()
    board += 1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241126233627674](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241126233627674.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

dfs和bfs的题目还是要花好长时间，第五题和第六题都做了好久

感觉考试要出这种题目好像可以直接缴械投降了





