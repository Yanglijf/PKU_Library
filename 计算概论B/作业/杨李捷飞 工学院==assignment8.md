# Assignment #8: 田忌赛马来了

Updated 1021 GMT+8 Nov 12, 2024

2024 fall, Complied by <mark>同学的姓名、院系</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 12558: 岛屿周⻓

matices, http://cs101.openjudge.cn/practice/12558/ 

思路：遍历每块陆地，它对周长的贡献是上下左右0的贡献

用时：10分钟左右

代码：

```python
n,m = map(int, input().split())
island = [[0]*(m+2)]
for _ in range(n):
    arr = list(map(int, input().split()))
    arr1 = [0] + arr[:] + [0]
    island.append(arr1)
island.append([0]*(m+2))
movements = [(-1,0),(1,0),(0,1),(0,-1)]
c = 0
for i in range(1,n+1):
    for j in range(1,m+1):
        if island[i][j] == 1:
            k = 0
            for x,y in movements:
                if island[i+x][j+y] == 0:
                    k += 1
            c += k
print(c)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![屏幕截图 2024-11-12 235005](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-11-12 235005.png)



### LeetCode54.螺旋矩阵

matrice, https://leetcode.cn/problems/spiral-matrix/

与OJ这个题目一样的 18106: 螺旋矩阵，http://cs101.openjudge.cn/practice/18106

思路：加保护圈，定义四个方向，然后按方向进行即可

用时：30分钟左右

代码：

```python
n = int(input())
l = [[-1]*(n+2)]
matrix = l + [[-1] + [0]*n + [-1] for _ in range(n)] + l
directions = [[0,1],[1,0],[0,-1],[-1,0]]
x,y = 1,1
dx,dy = directions[0]
i = 0
for k in range(1,n**2+1):
    matrix[x][y] = k
    if matrix[x+dx][y+dy] != 0:
        i += 1
        dx,dy = directions[i % 4]
    x += dx
    y += dy
for t in range(1,n+1):
    print(' '.join(map(str,matrix[t][1:n+1])))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241113130003373](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241113130003373.png)



### 04133:垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/

思路：遍历所有元素，实时更新投放点数目和清理数目

用时：25分钟左右

代码：

```python
d = int(input())
n = int(input())
x_list = []
y_list = []
num_list = []
for _ in range(n):
    x, y, num = map(int, input().split())
    x_list.append(x)
    y_list.append(y)
    num_list.append(num)
x_min = min(x_list)
x_max = max(x_list)
y_min = min(y_list)
y_max = max(y_list)
p_num = 0
r_num = 0
for i in range(max(0, x_min - d), min(1025, x_max + d + 1)):
    for j in range(max(0, y_min - d), min(1025, y_max + d + 1)):
        r = 0
        x_range = range(max(0, i - d), min(1025, i + d + 1))
        y_range = range(max(0, j - d), min(1025, j + d + 1))
        for t in range(n):
            if x_list[t] in x_range and y_list[t] in y_range:
                r += num_list[t]
        if r > r_num:
            p_num = 1
            r_num = r
        elif r == r_num:
            p_num += 1
print(str(p_num) + ' ' + str(r_num))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241113130412504](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241113130412504.png)



### LeetCode376.摆动序列

greedy, dp, https://leetcode.cn/problems/wiggle-subsequence/

与OJ这个题目一样的，26976:摆动序列, http://cs101.openjudge.cn/routine/26976/

思路：构建二维dp数组，分别储存最后是升或降的摆动序列的长度，最后输出二者最大值

用时：25分钟左右

代码：

```python
n = int(input())
nums = list(map(int, input().split()))
dp = [[1, 1] for _ in range(n)]
for i in range(1,n):
    if nums[i] > nums[i-1]:
        dp[i][1] = dp[i-1][0] + 1
        dp[i][0] = dp[i-1][0]
    elif nums[i-1] > nums[i]:
        dp[i][1] = dp[i-1][1]
        dp[i][0] = dp[i-1][1] + 1
    else:
        dp[i] = dp[i-1]
print(max(dp[n-1]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241113135641547](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241113135641547.png)



### CF455A: Boredom

dp, 1500, https://codeforces.com/contest/455/problem/A

思路：根据输入数据的最大值，构建count数组计算每个数字的数量，再构建dp数组计算最大数时的最大得分

用时：25分钟左右

代码：

```python
n = int(input())
nums = list(map(int,input().split()))
max_num = max(nums)
count = [0]*(max_num+1)
dp = [0]*(max_num+1)
for x in nums:
    count[x] += 1
for i in range(1,max_num+1):
    dp[i] = max(dp[i-1],dp[i-2] + i*count[i])
print(dp[max_num])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241113160205541](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241113160205541.png)



### 02287: Tian Ji -- The Horse Racing

greedy, dfs http://cs101.openjudge.cn/practice/02287

思路：按升序排列田和王的马，使用双指针，先对两端进行处理，若两端田大于王，则向内部缩进；若不然，用田最慢的马抵掉王最快的马（此时注意若相等则赢的次数不变）

用时：1小时

代码：

```python
while True:
    n = int(input())
    if n == 0:
        break
    tian = sorted(list(map(int ,input().split())))
    king = sorted(list(map(int ,input().split())))
    wins = 0
    tian_left, tian_right = 0, n-1
    king_left, king_right = 0, n-1
    while tian_left <= tian_right:
        if tian[tian_right] > king[king_right]:
            wins += 1
            tian_right -= 1
            king_right -= 1
        elif tian[tian_left] > king[king_left]:
            wins += 1
            tian_left += 1
            king_left += 1
        else:
            if tian[tian_left] < king[king_right]:
                tian_left += 1
                king_right -= 1
                wins -= 1
            else:
                tian_left += 1
                king_right -= 1
    print(wins*200)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241113173802325](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241113173802325.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

本次作业除田忌赛马想清楚思路、修改用时很久之外，得益于每日选做的及时跟进，其他题目思路还是较为顺畅

代码书写的熟练度还是需要提升



