# Assignment #5: Greedy穷举Implementation

Updated 1939 GMT+8 Oct 21, 2024

2024 fall, Complied by <mark>同学的姓名、院系</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### 04148: 生理周期

brute force, http://cs101.openjudge.cn/practice/04148

思路：用中国剩余定理算出日期

用时：20分钟左右

代码：

```python
n = 1
while True:
    p, e, i, d = map(int, input().split())
    if (p, e, i, d) == (-1, -1, -1, -1):
        break
    t = (5544 * p + 14421 * e + 1288 * i) % 21252
    if t <= d:
        t += 21252
    m = t - d
    print(f"Case {n}: the next triple peak occurs in {m} days.")
    n += 1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-22 160442.png)



### 18211: 军备竞赛

greedy, two pointers, http://cs101.openjudge.cn/practice/18211

思路：先进行排序，优先制作成本低的，卖出成本高的，然后双指针限定

用时：10分钟左右

代码：

```python
p = int(input())
l = list(map(int, input().split()))
l.sort()
s = 0
t = 0
m = 0
k = len(l)-1
for i in range(len(l)):
    if i > k:
        break
    elif p >= l[i]:
       s += 1
       m += 1
       p -= l[i]
    elif s > t:
        p += l[k]
        p -= l[i]
        k -= 1
        t += 1
        s += 1
print(m)
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-22 160723.png)



### 21554: 排队做实验

greedy, http://cs101.openjudge.cn/practice/21554

思路：给每个时间按顺序添加标号，按时间升序排列，然后输出标号，计算等待时间

用时：10分钟左右

代码：

```python
n = int(input())
time = list(map(int, input().split()))
time_number = []
for i in range(n):
    time_number.append((time[i],i+1))
time_number.sort()
wait_time = 0
for i in range(n):
    print(time_number[i][1],end=' ')
    wait_time += time_number[i][0]*(n-1-i)
wait_time /= n
print(f'\n{wait_time:.2f}')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-22 164051.png)



### 01008: Maya Calendar

implementation, http://cs101.openjudge.cn/practice/01008/

思路：将所有名称模相应周期进行对应，算出输入的天数，再转换成输出的日期格式

用时 ：15分钟左右

代码：

```python
dct1 = {'pop':0,'no':1,'zip':2,'zotz':3,'tzec':4,'xul':5,'yoxkin':6,'mol':7,'chen':8,'yax':9,'zac':10,'ceh':11,'mac':12,'kankin':13,'muan':14,'pax':15,'koyab':16,'cumhu':17,'uayet':18}
dct2 = {0:'imix',1:'ik',2:'akbal',3:'kan',4:'chicchan',5:'cimi',6:'manik',7:'lamat',8:'muluk',9:'ok',10:'chuen',11:'eb',12:'ben',13:'ix',14:'mem',15:'cib',16:'caban',17:'eznab',18:'canac',19:'ahau'}
n = int(input())
print(n)
for _ in range(n):
    day, month, year = input().split()
    day = int(day[:len(day)-1])
    year = int(year)
    days_num = 365*year + 20*dct1[month] + day + 1
    new_year = days_num//260
    days_num = days_num % 260
    if days_num == 0:
        days_num = 260
        new_year -= 1
    day_name = dct2[(days_num-1)%20]
    number = (days_num-1)%13 + 1
    print(str(number) + ' ' + day_name + ' ' + str(new_year))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-22 164444.png)



### 545C. Woodcutters

dp, greedy, 1500, https://codeforces.com/problemset/problem/545/C

思路：先对x坐标排序，从左开始，优先向左砍到，计算数量

用时：15分钟左右

代码：

```python
n = int(input())
x_h = []
for _ in range(n):
    x,h = map(int, input().split())
    x_h.append([x,h])
x_h.sort(key=lambda x:x[0])
x_h = [[-x_h[0][1],0]] + x_h[:] + [[x_h[n-1][0]+x_h[n-1][1]+1,0]]
result = 0
for i in range(1,n+1):
    if x_h[i][0] - x_h[i-1][0] > x_h[i][1]:
        result += 1
    elif x_h[i+1][0] - x_h[i][0] > x_h[i][1]:
        result += 1
        x_h[i][0] += x_h[i][1]
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### 01328: Radar Installation

greedy, http://cs101.openjudge.cn/practice/01328/

思路：按x坐标进行小岛排序，从左到右判断是否在fanwein，若不在，雷达数加一，坐标为覆盖该小岛的上确界

用时：40分钟左右

代码：

```python
import math
case_number=0
while True:
    n,d=map(int,input().split())
    if n==0 and d==0:
        break
    else :
        case_number+=1
        rge = []
        b = 0
        for _ in range(n):
            x,y=map(int,input().split())
            if y > d or y < 0:
                b = 1
            else:
                r = math.sqrt(d**2-y**2)
                rge.append([x-r,x+r])
        if b==0:
            rge.sort(key=lambda t:t[0])
            c=rge[0][1]
            result=1
            for j in range(1,n):
                if rge[j][0]>c:
                    result+=1
                    c=rge[j][1]
                elif rge[j][1]<c:
                    c=rge[j][1]
            print(f'Case {case_number}: {result}')
        if b==1:
            print(f'Case {case_number}: {-1}')
        k=input()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Pictures\Screenshots\屏幕截图 2024-10-22 164851.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

贪心问题首先需要想清楚每一步的操作，并及时更新边界

最好可以通过数学方法进行优化，尽量在一次循环解决问题

明显感觉最近几天的每日选做难度加大，优化的思路还是不太好想，还是要继续做题



