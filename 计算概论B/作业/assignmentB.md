# Assignment #B: Dec Mock Exam大雪前一天

Updated 1649 GMT+8 Dec 5, 2024

2024 fall, Complied by <mark>同学的姓名、院系</mark>



**说明：**

1）⽉考： AC1<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### E22548: 机智的股民老张

http://cs101.openjudge.cn/practice/22548/

思路：建两个列表，储存以第i天为分界时买的最低价，卖的最高价，然后做差

用时：15分钟

代码：

```python
a = list(map(int,input().split()))
n = len(a)
l1 = [a[0]]
for i in range(1,n):
    l1.append(min(a[i],l1[-1]))
l2 = [a[n-1]]
for i in range(n-2,-1,-1):
    l2.append(max(a[i],l2[-1]))
l2 = l2[::-1]
result = 0
for i in range(n):
    result = max(result,l2[i]-l1[i])
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241206095443659](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241206095443659.png)



### M28701: 炸鸡排

greedy, http://cs101.openjudge.cn/practice/28701/

思路：总时间除以k是时间的上届，大于这个时间的鸡排直接扔进锅里即可，然后划归到k-1情形

用时：50分钟

代码：

```python
n, k = map(int, input().split())
time = list(map(int, input().split()))
time.sort()
total_time = sum(time)
while time:
    if time[-1] > total_time/k:
        total_time -= time.pop()
        k -= 1
    else:
        print(f'{total_time/k:.3f}')
        break
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20241206122722949](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241206122722949.png)



### M20744: 土豪购物

dp, http://cs101.openjudge.cn/practice/20744/

思路：构建两个dp数组，一个储存不放回的极大值，一个储存可以有放回的极大值，通过递推取第二个的最值

用时：1小时

代码：

```python
value = list(map(int, input().split(',')))
n = len(value)
dp1 = [0]*n
dp2 = [0]*n
result = dp1[0] = dp2[0] = value[0]
for i in range(1,n):
    dp1[i] = max(dp1[i-1] + value[i],value[i])
    dp2[i] = max(dp1[i-1],dp2[i-1] + value[i])
    result = max(dp2[i],result)
print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241206095837597](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241206095837597.png)



### T25561: 2022决战双十一

brute force, dfs, http://cs101.openjudge.cn/practice/25561/

思路：先储存商品和商店折扣，dfs遍历所有（数据量较小）购买种类，然后分别按照折扣计算总价并取最小值

用时：40分钟

代码：

```python
def dfs(product_note,buy_plan,products,discounts):
    global min_cost

    if product_note == n:
        total = 0
        store_costs = {}
        for product,store,price in buy_plan:
            if store not in store_costs:
                store_costs[store] = 0
            store_costs[store] += price
            total += price
        final = 0
        for store,cost in store_costs.items():
            discount = 0
            if store in discounts:
                for q,x in discounts[store]:
                    if cost >= q:
                        discount = max(discount,x)
            final += cost - discount
        all_discount = (total//300)*50
        final -= all_discount

        min_cost = min(min_cost,final)

        return

    for store,price in products[product_note]:
        buy_plan.append((product_note,store,price))
        dfs(product_note+1,buy_plan,products,discounts)
        buy_plan.pop()


n,m = map(int, input().split())

products = []
for _ in range(n):
    l = input().split()
    prices = []
    for x in l:
        store,price = map(int,x.split(':'))
        prices.append((store,price))
    products.append(prices)

discounts = {}
for i in range(1,m+1):
    l = input().split()
    store = []
    for xx in l:
        q,x = map(int, xx.split('-'))
        store.append((q,x))
    discounts[i] = store

min_cost = float('inf')
dfs(0,[],products,discounts)
print(min_cost)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Downloads\屏幕截图_6-12-2024_122634_cs101.openjudge.cn.jpeg)



### T20741: 两座孤岛最短距离

dfs, bfs, http://cs101.openjudge.cn/practice/20741/

思路：先dfs分别标记并储存两岛的坐标，然后bfs从一个出发计算最短距离

用时：40分钟

代码：

```python
from collections import deque

movements = [[1, 0], [-1, 0], [0, 1], [0, -1]]
l1 = []
l2 = []
def denote(ll):
    islands = [[0]*n for _ in range(n)]
    count = 0

    def dfs(x,y,note):
        global l1,l2
        if x < 0 or x >= n or y < 0 or y >= n or ll[x][y] == '0' or islands[x][y] != 0:
            return
        islands[x][y] = note
        if note == 1:
            l1.append((x,y))
        else:
            l2.append((x,y))
        for i in range(4):
            dfs(x+movements[i][0],y+movements[i][1],note)

    for x in range(n):
        for y in range(n):
            if ll[x][y] == '1' and islands[x][y] == 0:
                count += 1
                dfs(x,y,count)

def min_distance(l1,l2,ll):
    queue = deque(l1)
    visited = set(l1)
    distance= 0
    while queue:
        lth = len(queue)
        for _ in range(lth):
            x,y = queue.popleft()
            if (x,y) in l2:
                return distance - 1

            for dx,dy in movements:
                nx,ny = x+dx,y+dy
                if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        distance += 1
    return -1


n = int(input())
ll = [input() for _ in range(n)]
denote(ll)
print(min_distance(l1,l2,ll))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\yangljf\Downloads\屏幕截图_6-12-2024_1005_cs101.openjudge.cn.jpeg)



### T28776: 国王游戏

greedy, http://cs101.openjudge.cn/practice/28776

思路：大臣获得的金币是自己以及前面所有左手乘积除以自己的左右手乘积，从这个思路出发按左右手乘积升序排列即可获得最大值的最小值，且该值在最后一个大臣处取到（可详细论证）

用时：45分钟

代码：

```python
n = int(input())
a,b = map(int, input().split())

pre = [0]*(n+1)
pre[0] = a

l = []
for _ in range(n):
    a,b = map(int, input().split())
    l.append((a,b))
l.sort(key=lambda x:x[0]*x[1])

for i in range(1,n+1):
    pre[i] = pre[i-1] * l[i-1][0]

print(pre[n]//(l[-1][0]*l[-1][1]))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241206102826015](C:\Users\yangljf\AppData\Roaming\Typora\typora-user-images\image-20241206102826015.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

月考考的最差的一次，第一题卡住了一会儿，打乱了后面做题的节奏，不能第一时间确定算法，代码复杂的话思维也容易混乱

还是需要加练题目，提高算法的熟练度



