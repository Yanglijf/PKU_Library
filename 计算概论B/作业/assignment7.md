# Assignment #7: Nov Mock Exam立冬

Updated 1646 GMT+8 Nov 7, 2024

2024 fall, Complied by <mark>同学的姓名、院系</mark>



**说明：**

1）⽉考： AC6<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E07618: 病人排队

sorttings, http://cs101.openjudge.cn/practice/07618/

思路：将老年人和年轻人分别储存，按照各自的顺序输出

用时：10分钟左右

代码：

```python
n=int(input())
l1 = []
l2 = []
for _ in range(n):
    a,b = input().split()
    if int(b) >= 60:
        l1.append((a,int(b)))
    else:
        l2.append((a,b))
l1.sort(key=lambda x:x[1],reverse=True)
for x in l1:
    print(x[0])
for x in l2:
    print(x[0])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-11-07 182003.png)



### E23555: 节省存储的矩阵乘法

implementation, matrices, http://cs101.openjudge.cn/practice/23555/

思路：先将三元组转换成矩阵，然后进行矩阵乘法并输出非零元

用时：15分钟左右

代码：

```python
n,m1,m2 = map(int, input().split())
A = [[0 for i in range(n)] for j in range(n)]
B = [[0 for i in range(n)] for j in range(n)]
for _ in range(m1):
    x,y,v = map(int, input().split())
    A[x][y] = v
for _ in range(m2):
    x,y,v = map(int, input().split())
    B[x][y] = v
for i in range(n):
    for j in range(n):
        t = 0
        for k in range(n):
            t += A[i][k]*B[k][j]
        if t!= 0:
            print(f"{i} {j} {t}")
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-11-07 182209.png)



### M18182: 打怪兽 

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/

思路：将时刻于技能组成的列表对应起来，然后计算m个技能的最大伤害并更新键值，然后按时间排序并计算

用时：15分钟左右

代码：

```python
t = int(input())
for _ in range(t):
    n,m,b = map(int, input().split())
    arr = {}
    for i in range(n):
        t,x = map(int,input().split())
        if t in arr:
            arr[t].append(x)
        else:
            arr[t] = [x]
    for x in arr:
        arr[x].sort(reverse=True)
        arr[x] = sum(arr[x][0:m])
    arr1 = sorted(arr)
    for t in arr1:
        b -= arr[t]
        if b <= 0:
            print(t)
            break
    else:
        print("alive")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-11-07 182439.png)



### M28780: 零钱兑换3

dp, http://cs101.openjudge.cn/practice/28780/

思路：使用dp并且遍历所有硬币

用时：15分钟

代码：

```python
n, m = map(int, input().split())
arr = list(map(int,input().split()))
dp = [float('inf')]*(m+1)
dp[0] = 0
for i in range(1,m+1):
    for x in arr:
        if i >= x:
            dp[i] = min(dp[i], dp[i-x]+1)
if dp[m] <= m+1:
    print(dp[m])
else:
    print('-1')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-11-07 200651.png)



### T12757: 阿尔法星人翻译官

implementation, http://cs101.openjudge.cn/practice/12757

思路：首先进行键值对应，将hundred，thousand，million分开处理

用时：45分钟

代码：

```python
s = input().split()
keys = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million' ]
values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100,1000,1000000]
dct = dict(zip(keys,values))
if s[0] == 'negative':
    s1 = s[1:]
    result = 0
    current = 0
    for i in s1:
        if i in ['thousand', 'million']:
            current *= dct[i]
            result += current
            current = 0
            continue
        if i == 'hundred':
            current *= dct[i]
        else:
            current += dct[i]
    print((-1)*(result+current))
else:
    result = 0
    current = 0
    for i in s:
        if i in ['thousand', 'million']:
            current *= dct[i]
            result += current
            current = 0
            continue
        if i == 'hundred':
            current *= dct[i]
        else:
            current += dct[i]
    print(result + current)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-11-07 201543.png)



### T16528: 充实的寒假生活

greedy/dp, cs10117 Final Exam, http://cs101.openjudge.cn/practice/16528/

思路：按结束时间排序，并使用贪心算法得出最优解

用时：20分钟左右

代码：

```python
n = int(input())
act = []
for _ in range(n):
    a,b = map(int,input().split())
    act.append([a,b])
act.sort(key=lambda  x:x[1])
result = 1
lim = act[0][1]
for i in range(1,n):
    if act[i][0] > lim:
        result += 1
        lim = act[i][1]
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-11-07 201229.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

这次考试还是在动态规划和dp题目上卡了较长时间，第五题在把hundred和thousand，million分开处理时思考了较长时间，最后没时间提交

这两周准备其他科目的期中考试，每日选做跟进不是很好，以后还是要大量做题



