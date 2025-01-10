# Assignment #6: Recursion and DP

Updated 2201 GMT+8 Oct 29, 2024

2024 fall, Complied by <mark>同学的姓名、院系</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### sy119: 汉诺塔

recursion, https://sunnywhy.com/sfbj/4/3/119  

思路：首先可以计算出结果为2^n - 1,然后递归的定义函数即可

用时：10分钟

代码：

```python
def remove_ABC(n,p1,p2,p3):
    if n == 1:
        print(f'{p1}->{p3}')
    else:
        remove_ABC(n-1,p1,p3,p2)
        print(f'{p1}->{p3}')
        remove_ABC(n-1,p2,p1,p3)
n = int(input())
ans = 2**n - 1
print(ans)
remove_ABC(n,'A','B','C')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-30 131107.png)



### sy132: 全排列I

recursion, https://sunnywhy.com/sfbj/4/3/132

思路：第一遍为使用dfs导致没有按照字典序输出，使用dfs后可以保证字典序

用时：30分钟

代码：

```python
n = int(input())
perms = []
used = [False]*(n+1)
def permute(lth,n,used,current_perm,perms):
    if lth == n:
        perms.append(current_perm[:])
    else:
        for i in range(1,n+1):
            if not used[i]:
                current_perm.append(i)
                used[i] = True
                permute(lth+1,n,used,current_perm,perms)
                used[i] = False
                current_perm.pop()
permute(0,n,used,[],perms)
for perm in perms:
    print(' '.join(map(str, perm)))
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-31 102917.png)



### 02945: 拦截导弹 

dp, http://cs101.openjudge.cn/2024fallroutine/02945

思路：递归的记录每一个有序子列的长度，最后输出其最大值

用时：10分钟

代码：

```python
k = int(input())
missiles = list(map(int ,input().split()))
dp = [1]*k
for i in range(1,k):
    for j in range(i):
        if missiles[j] >= missiles[i]:
            dp[i] = max(dp[i], dp[j] + 1)
print(max(dp))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-30 134856.png)



### 23421: 小偷背包 

dp, http://cs101.openjudge.cn/practice/23421

思路：构造二维dp数组，表示收纳前i个物体，占据j个重量的最大价值

用时：30分钟

代码：

```python
n, b = map(int, input().split())
values = list(map(int, input().split()))
weights = list(map(int, input().split()))
dp = [[0]*(b+1) for _ in range(n+1)]
for i in range(1,n+1):
    for j in range(b+1):
        if weights[i - 1] <= j:
            dp[i][j] = max(dp[i - 1][j],dp[i - 1][j - weights[i - 1]] + values[i - 1])
        else:
            dp[i][j] = dp[i - 1][j]
print(dp[n][b])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-30 213210.png)



### 02754: 八皇后

dfs and similar, http://cs101.openjudge.cn/practice/02754

思路：进行函数迭代，判断当前位置是否在以前位置的同一行或斜线上，如果都不在，进行下一函数

用时：30分钟

代码：

```python
queen = []
def q(s):
    for i in range(1,9):
        for j in range(len(s)):
            if str(i) == s[j] or abs(i-int(s[j])) == abs(len(s)-j):
                break
        else:
            if len(s) == 7:
                queen.append(s + str(i))
            else:
                q(s + str(i))
q('')
n = int(input())
for _ in range(n):
    x = int(input())
    print(queen[x-1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-30 221516.png)



### 189A. Cut Ribbon 

brute force, dp 1300 https://codeforces.com/problemset/problem/189/A

思路：递归的判断每段长度可以剪的最大数量

用时：15分钟

代码：

```python
n, a, b, c = map(int,input().split())
A = [-1]*(n+1)
A[0] = 0
for i in range(1, n+1):
    if i >= a and A[i-a] != -1:
        A[i] = max(A[i],A[i-a]+1)
    if i >= b and A[i-b] != -1:
        A[i] = max(A[i],A[i-b]+1)
    if i >= c and A[i-c] != -1:
        A[i] = max(A[i],A[i-c]+1)
print(A[n])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-30 221752.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

本次作业递归题目思路比较难想，经常会花费很长时间，但想明白后函数反而比较好写

dp题目中dp的构造也需要一定时间的斟酌

目前还是要跟上每日选做的节奏，打开自己的思维



