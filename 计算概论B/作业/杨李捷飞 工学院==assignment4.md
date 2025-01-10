# Assignment #4: T-primes + 贪心

Updated 0337 GMT+8 Oct 15, 2024

2024 fall, Complied by <mark>同学的姓名、院系</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知9月19日导入选课名单后启用。**作业写好后，保留在自己手中，待9月20日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 34B. Sale

greedy, sorting, 900, https://codeforces.com/problemset/problem/34/B



思路：在限定数量的前提下选择绝对值大的负数，相加取绝对值

用时：5分钟左右

代码

```python
n, m = map(int, input().split())
l = list(map(int, input().split()))
l.sort()
k = 0
w = 0
for x in l:
    if x < 0 and k < m:
        w += x
        k += 1
    else:
        break
print(-w)
```

（python）

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-16 175710.png)



### 160A. Twins

greedy, sortings, 900, https://codeforces.com/problemset/problem/160/A

思路：从面额较大的硬币开始选，价值超过一般总值时停止，输出硬币数量

用时：5分钟左右

代码

```python
n = int(input())
l = list(map(int, input().split()))
l.sort(reverse=True)
l.append(0)
k = sum(l)
r = 0
w = 0
for x in l:
    if w <= k/2:
        w += x
        r += 1
    else:
        print(r)
        break

```

（python）

代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-16 175938.png)



### 1879B. Chips on the Board

constructive algorithms, greedy, 900, https://codeforces.com/problemset/problem/1879/B

思路：最优策略是选择a或b中的最小数，与b或a中元素总和相加，取最小值

用时：15分钟左右

代码

```python
t = int(input())
for _ in range(t):
    n = int(input())
    a = sorted(list(map(int, input().split())))
    b = sorted(map(int, input().split()))
    w2 = sum(a) + n*b[0]
    w3 = sum(b) + n*a[0]
    print(min(w2,w3))

```

（python）

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-16 180215.png)



### 158B. Taxi

*special problem, greedy, implementation, 1100, https://codeforces.com/problemset/problem/158/B

思路：从小组人数为4开始排，再3，再1，再2，再1，所用车数即为最小

用时：10分钟左右

代码

```python
import math
n = int(input())
s = list(map(int, input().split()))
a = s.count(1)
b = s.count(2)
c = s.count(3)
d = s.count(4)
if b % 2 == 0:
    if c < a:
        m = d + b/2 + c + math.ceil((a-c)/4)
    else:
        m = d + b/2 + c
else:
    if a <= 2:
        m = d + (b+1)/2 + c
    else:
        a = a -2
        if c < a:
            m = d + (b+1)/2 + c + math.ceil((a-c)/4)
        else:
            m = d + (b+1)/2 + c
print(int(m))

```

（python）

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-16 180453.png)



### *230B. T-primes（选做）

binary search, implementation, math, number theory, 1300, http://codeforces.com/problemset/problem/230/B

思路：先预算出10**6以内的所有素数，然后判断输入的数字是否为素数的平方

用时：1小时左右

代码

```python
import math
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return is_prime
limit = 10**6
is_prime = sieve(limit)
n = int(input())
numbers = map(int, input().split())
for x in numbers:
    root = int(math.isqrt(x))
    if root * root != x:
        print('NO')
    elif root <= limit and is_prime[root]:
        print('YES')
    else:
        print('NO')
```

（python）

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-16 180853.png)



### *12559: 最大最小整数 （选做）

greedy, strings, sortings, http://cs101.openjudge.cn/practice/12559

思路：对字符串进行倍长，倍数为最长与最短的比的上取整（本题限定在1000以内，故取倍数为3即可），然后进行排序，拼接

用时：40分钟

代码

```python
n = int(input())
l = list(map(str, input().split()))
d = []
for x in l:
    d.append(len(x))
x = min(d)
y = max(d)
t = y//x + 1
l.sort(key=lambda x: x*t, reverse=True)
print(''.join(l), end=' ')
l = l[::-1]
print(''.join(l))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-16 181211.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

对于贪心算法的循环的每一步都要想清楚，避免影响下一循环

对于复杂题目变量名称尽量清楚易懂，避免意义不明引起混乱

关注输入数据的范围或许可以帮助简化代码



