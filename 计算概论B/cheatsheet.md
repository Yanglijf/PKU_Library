#### 常用库/函数

##### **1. 核心功能 :**

`int()`:  整数。`float()`:  浮点数。`str()`:  字符串。`bool(x)`: 布尔值。`list()`: 列表。`tuple()`: 元组。`set()`: 集合。`dict()`: 字典。

`abs()`: 绝对值。`pow(x, y)`: x 的 y 次方。`round(x, n)`: x 四舍五入到 n 位小数。`max()`: 最大值。`min()`: 最小值。`sum()`: 和。`len()`: 长度。`divmod(x,y)` :返回 `(x//y, x%y)`，即商和余数。

`sorted()`:排序。`reversed()`:反转。`range(start, stop, step)`: 整数序列。`enumerate()`: 包含索引和值。`zip(, , ...)`: 多个元素的元组.`all()`:如果序列的所有元素为真，返回 `True`。`any()`:如果序列中有任意一个元素为真，返回 `True`。`map()`:对序列的每个元素应用一个函数。`''.join()`:连接字符串。

`find(sub,start,end)` `count(sub,start,end)`

`chr()` :Unicode 字符。`ord()`:对应的 Unicode 码点。`ascii()`:ASCII 表示。`repr()`:字符串表示（带引号）。

`global`: 全局变量。`nonlocal`:外层函数变量。

##### **2. 常用模块:**

###### `heapq`模块

1.**`heapq.heappush(heap, item)`**2.**`heapq.heappop(heap)`**

3.**`heapq.heappushpop(heap, item)`**4.**`heapq.heapreplace(heap, item)`**

5.**`heapq.heapify(heap)`**6.**`heapq.nlargest(n, iterable, key=None)`**

7.**`heapq.nsmallest(n, iterable, key=None)`**

**综合示例：求数据流的中位数**

```python
import heapq
class MedianFinder:
    def __init__(self):
        self.small = []  # 最大堆（存储较小的一半，取负数）
        self.large = []  # 最小堆（存储较大的一半）
    def addNum(self, num):
        # 将新数字加入最大堆
        heapq.heappush(self.small, -num)
        # 平衡最大堆和最小堆
        heapq.heappush(self.large, -heapq.heappop(self.small))
        # 如果最小堆的元素多于最大堆，平衡两边
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2

```

###### `sys` 模块 `sys.setrecursionlimit(limit)`: 用于获取和设置 Python 解释器的最大递归深度。

###### `math`模块

向上取整和向下取整`math.ceil()` `math.floor()` 幂运算`math.pow(, )`平方根`math.sqrt()` 

对数函数：自然对数`math.log(math.e)` 以10为底的对数`math.log10()`

以指定底数的对数`math.log(8, 2)`    返回3.0 (以2为底8的对数)

阶乘`math.factorial()` 最大公约数`math.gcd()`

###### `collections` 模块

1.**`collections.deque`**

```python
from collections import deque
dq.append(4) dq.appendleft(0) dq.pop() dq.popleft()
dq = deque(maxlen=3) dq.extend([1, 2, 3]) dq.append(4)  # 超过长度时，左端的元素会被自动移除
```

2.**`collections.Counter`**

```python
from collections import Counter
cnt = Counter("abracadabra")
print(cnt)  # 输出: Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
```

3.**`collections.defaultdict`** `int` `set` `list`

###### `itertools` 模块

1.**排列组合**

- **`itertools.permutations(iterable, r=None)`**  
  返回 `iterable` 中所有长度为 `r` 的排列。如果 `r` 未指定，则默认为 `iterable` 的长度。

  ```python
  import itertools
  print(list(itertools.permutations('ABC', 2)))
  ```
  
- **`itertools.combinations(iterable, r)`**  
  返回 `iterable` 中所有长度为 `r` 的组合（不允许重复选择，顺序无关）。

  ```python
  import itertools
  print(list(itertools.combinations('ABC', 2)))
  ```
  
- **`itertools.combinations_with_replacement(iterable, r)`**  
  返回 `iterable` 中所有长度为 `r` 的组合（允许重复选择）。

  ```python
  import itertools
  print(list(itertools.combinations_with_replacement('ABC', 2)))
  ```

2.**组合多个迭代器的函数**

- **`itertools.product(*iterables, repeat=1)`**  返回多个可迭代对象的笛卡尔积。
  
  ```python
  import itertools
  print(list(itertools.product('AB', '12')))
  ```

3.**累积计算**

**`itertools.accumulate(iterable, func=operator.add)`**  。

```python
import itertools
print(list(itertools.accumulate([1, 2, 3, 4])))
```

使用乘法来累积

```python
import itertools
import operator
result = list(itertools.accumulate([1, 2, 3, 4, 5], operator.mul))
print(result)
```

 使用 `max` 函数

```python
import itertools
result = list(itertools.accumulate([3, 1, 4, 1, 5, 9, 2, 6], max))
print(result)
```

 自定义函数

```python
import itertools
def diff(x, y):
    return x - y
result = list(itertools.accumulate([10, 1, 2, 3, 4], diff))
print(result)
```

计算模累积

```python
import itertools
result = list(itertools.accumulate([1, 2, 3, 4, 5], lambda x, y: (x + y) % 3))
print(result)
```

累积计算字符串连接

```python
import itertools
result = list(itertools.accumulate(["a", "b", "c", "d"], lambda x, y: x + y))
print(result)
```

使用初始值 (`initial` 参数)

```python
import itertools
result = list(itertools.accumulate([1, 2, 3, 4], initial=10))
print(result)
```

###### `functools`模块

```python
import functools
@functools.lru_cache(maxsize=)  
```



#### 二分查找

##### 模板

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1  # 定义搜索范围的左边界和右边界
    while left <= right:  # 当左边界小于或等于右边界时继续搜索
        mid = (left + right) // 2  # 计算中间位置
        if arr[mid] == target:  # 如果找到目标值，返回索引
            return mid
        elif arr[mid] < target:  # 如果中间值小于目标值，缩小范围到右半部分
            left = mid + 1
        else:  # 如果中间值大于目标值，缩小范围到左半部分
            right = mid - 1
    return -1  # 如果未找到目标值，返回 -1
```

##### **核心功能：**

**`bisect.bisect_left(a, x, lo=0, hi=len(a))` 或 `bisect.bisect(a, x, lo=0, hi=len(a))`**

**`bisect.bisect_right(a, x, lo=0, hi=len(a))`**

**`bisect.insort_left(a, x, lo=0, hi=len(a))`**

**`bisect.insort_right(a, x, lo=0, hi=len(a))`  `bisect.insort(a, x, lo=0, hi=len(a))`**

#### 二分归并

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr  # 递归终止条件：当子序列只有一个元素时，它本身就是有序的
    mid = len(arr) // 2  # 找到中间位置
    left_half = arr[:mid]  # 左子序列
    right_half = arr[mid:]  # 右子序列
    # 递归地对左右子序列进行归并排序
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    # 合并两个已排序的子序列
    return merge(left_half, right_half)
def merge(left, right):
    merged = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
    # 将剩余的元素添加到合并后的序列中
    merged.extend(left[left_index:])
    merged.extend(right[right_index:])
    return merged
```

#### 素数筛选

##### 1.埃拉托斯特尼筛法（Sieve of Eratosthenes）通过标记合数来找到素数  O(n log log n)

```python
def sieve_of_eratosthenes(limit):
  is_prime = [True] * limit
  is_prime[0] = is_prime[1] = False  # 0 和 1 不是素数
  for p in range(2, int(math.sqrt(limit)) + 1):
    if is_prime[p]:
      for i in range(p * p, limit, p):
        is_prime[i] = False
  primes = [i for i, prime in enumerate(is_prime) if prime]
  return primes
```

##### **2. 优化后的埃拉托斯特尼筛法**

*   **只考虑奇数:** 除了 2 之外，所有的素数都是奇数，所以可以只筛选奇数。
*   **从 p*p 开始标记:** 对于一个素数 p，它的倍数 2p, 3p, ... 已经在之前的筛选中被标记过了，所以可以从 p*p 开始标记。

```python
def sieve_of_eratosthenes_optimized(limit):
    if limit <= 2:
        return []
    is_prime = [True] * (limit // 2)  # 只存储奇数的素数信息
    for p in range(3, int(math.sqrt(limit)) + 1, 2):
        if is_prime[p // 2]:
            for i in range(p * p, limit, 2 * p):
                if i % 2 != 0:
                    is_prime[i // 2] = False
    primes = [2] + [2 * i + 1 for i, prime in enumerate(is_prime[1:]) if prime]
    return primes
```

##### 3.快速判断（大数）

```python
def a(x):
    if x <= 1:
        return 0
    if x == 2 or x == 3:
        return 1
    if x % 2 == 0 or x % 3 == 0:
        return 0
    k = 5
    while k * k <= x:
        if x % k == 0:
            return 0
        k += 6
    return 1
```



#### 保留小数

##### **(1) f-string 格式化**
```python
num = 3.14159
result = f"{num:.2f}"  # 保留 2 位小数
print(result)  # 输出: 3.14
```

##### **(2) `format()` 方法**
```python
num = 3.14159
result = "{:.2f}".format(num)  # 保留 2 位小数
print(result)  # 输出: 3.14
```

### DFS

```python
def dfs_recursive(graph, start):
    """
    :param graph: 图的邻接表表示
    :param start: 起始节点
    """
    visited = set()  # 记录访问过的节点

    def dfs(node):
        visited.add(node)
        print(node, end=" ")  # 访问节点
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)

# 示例图
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

dfs_recursive(graph, 'A')
```

```python
def dfs_stack(graph, start):
    """
    :param graph: 图的邻接表表示
    :param start: 起始节点
    """
    visited = set()  # 记录访问过的节点
    stack = [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node, end=" ")  # 访问节点
            for neighbor in reversed(graph[node]):  # 注意这里使用reversed
                if neighbor not in visited:
                    stack.append(neighbor)

# 示例图
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

dfs_stack(graph, 'A')
```

### BFS

1. 基础 BFS 模板（使用队列）：
```python
from collections import deque

def bfs(start, target):
    queue = deque([start])  # 创建队列并加入起点
    visited = {start}       # 记录已访问的节点
    
    step = 0  # 记录步数（如果需要）
    
    while queue:
        size = len(queue)  # 当前层的节点数
        
        # 遍历当前层的所有节点
        for _ in range(size):
            cur = queue.popleft()  # 取出队首节点
            
            # 判断是否到达目标
            if cur == target:
                return step
            
            # 遍历相邻节点
            for next_node in get_neighbors(cur):
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append(next_node)
        
        step += 1
    
    return -1  # 未找到目标
```

2. 二维矩阵 BFS 模板：
```python
from collections import deque

def bfs_matrix(matrix, start_x, start_y):
    if not matrix or not matrix[0]:
        return
    
    m, n = len(matrix), len(matrix[0])
    queue = deque([(start_x, start_y)])
    visited = {(start_x, start_y)}
    
    # 四个方向：上、右、下、左
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while queue:
        cur_x, cur_y = queue.popleft()
        
        # 遍历四个方向
        for dx, dy in directions:
            next_x, next_y = cur_x + dx, cur_y + dy
            
            # 检查是否越界或已访问
            if (0 <= next_x < m and 0 <= next_y < n and 
                (next_x, next_y) not in visited):
                
                # 其他条件判断（根据具体问题）
                if matrix[next_x][next_y] == target:
                    # 处理逻辑
                    pass
                
                visited.add((next_x, next_y))
                queue.append((next_x, next_y))
```

3. 带层次的 BFS 模板：
```python
from collections import deque

def level_bfs(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        size = len(queue)
        
        # 处理当前层的所有节点
        for _ in range(size):
            node = queue.popleft()
            level.append(node.val)
            
            # 将下一层的节点加入队列
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

4. 双向 BFS 模板（适用于起点和终点都已知的情况）：
```python
from collections import deque

def bidirectional_bfs(start, target):
    if start == target:
        return 0
    
    # 从起点和终点开始的两个集合
    start_set = {start}
    end_set = {target}
    visited = set()
    
    step = 0
    
    while start_set and end_set:
        # 总是扩展较小的集合
        if len(start_set) > len(end_set):
            start_set, end_set = end_set, start_set
        
        next_level = set()
        
        # 扩展当前层的所有节点
        for cur in start_set:
            for next_node in get_neighbors(cur):
                if next_node in end_set:
                    return step + 1
                
                if next_node not in visited:
                    visited.add(next_node)
                    next_level.add(next_node)
        
        start_set = next_level
        step += 1
    
    return -1
```

#### 全排列

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

### Dijkstra算法

```python
import heapq
def dijkstra(graph, start):
    """
    使用 Dijkstra 算法计算从起点到其他所有节点的最短路径。
        :param graph: 图的邻接表表示，格式为 {节点: [(邻居, 权重), ...]}
    :param start: 起点
    :return: 从起点到每个节点的最短距离，格式为 {节点: 距离}
    """
    # 最短路径字典，初始值为无穷大
    distances = {node: float('inf') for node in graph}
    distances[start] = 0  # 起点到自身的距离为0
    # 最小堆，存储 (当前距离, 节点)
    pq = [(0, start)]  # 起始距离为0
    while pq:
        # 弹出堆顶节点
        current_distance, current_node = heapq.heappop(pq)
        # 如果弹出的距离大于当前记录的距离，跳过（因为可能是旧的、更长的路径）
        if current_distance > distances[current_node]:
            continue
        # 遍历当前节点的邻居
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            # 如果找到更短路径，则更新
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances
# 图的定义，使用邻接表表示
# 每个节点的值是一个列表，列表中的元素是 (邻居, 权重) 的元组
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 6)],
    'C': [('A', 4), ('B', 2), ('D', 3)],
    'D': [('B', 6), ('C', 3)]
}
# 起点
start_node = 'A'
# 运行 Dijkstra 算法
shortest_distances = dijkstra(graph, start_node)
# 输出结果
print("从起点 {} 到其他各点的最短距离:".format(start_node))
for node, distance in shortest_distances.items():
    print("{}: {}".format(node, distance))
```

#### 走山路（Dijkstra）

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

#### 建筑修建

```python
n, m = map(int, input().split())
x_y = []
for i in range(n):
    x, y = map(int, input().split())
    if i == 0:
        x_y.append((x,y))
    elif x == x_y[-1][0]:
        if y < x_y[-1][1]:
            x_y.pop()
            x_y.append((x,y))
    else:
        x_y.append((x, y))
s_e = []
for x,y in x_y:
    for start in range(max(0,x-y+1),min(m-y,x)+1):
        s_e.append((start,start+y))
s_e.sort(key=lambda x:(x[1],x[0]))

count = 0
current = 0
for start,end in s_e:
    if start >= current:
        current = end
        count += 1
print(count)
```

### 马拉车算法

马拉车算法是一种用于在 O(n) 时间复杂度内找到一个字符串中最长回文子串的算法。它巧妙地利用了回文串的对称性，避免了重复计算，从而提高了效率。

**核心思想**

1. **预处理字符串：** 为了处理奇数和偶数长度的回文串，我们在原始字符串的每个字符之间插入一个特殊字符（例如 `#`），并在字符串的首尾也添加特殊字符。 例如，`"aba"` 变为 `"#a#b#a#"`，`"abba"` 变为 `"#a#b#b#a#"`。
2. **维护一个辅助数组 `p`：** `p[i]` 表示以预处理后字符串 `s` 的第 `i` 个字符为中心的回文串的半径（包括中心字符本身）。例如，对于 `"#a#b#a#"`，`p[3]` 为 3，表示以 `b` 为中心的回文串 `#a#b#a#` 的半径为 3。
3. **利用回文串的对称性：** 在计算 `p[i]` 时，如果 `i` 在之前计算过的回文串的覆盖范围内，我们可以利用对称性快速计算出一个初始值，从而避免重复计算。
4. **动态维护最右边界和中心：** 在遍历过程中，我们需要维护当前已知的最右回文串的右边界 `mx` 和中心 `id`。

```python
def manacher(s):
    """
    马拉车算法，查找字符串中最长回文子串
    Args:
        s: 原始字符串
    Returns:
        最长回文子串
    """
    # 1. 预处理字符串
    t = "#" + "#".join(list(s)) + "#"
    n = len(t)
    p = [0] * n  # 辅助数组，p[i] 表示以 t[i] 为中心的回文半径
    mx = 0  # 当前已知的最右回文串的右边界
    id = 0  # 当前已知的最右回文串的中心
    max_len = 0  # 最长回文子串的半径
    center_index = 0 # 最长回文子串的中心索引
    for i in range(1, n):
        # 利用对称性，计算 p[i] 的初始值
        if i < mx:
            p[i] = min(p[2 * id - i], mx - i)
        else:
            p[i] = 1
        # 尝试扩展回文串
        while i - p[i] >= 0 and i + p[i] < n and t[i - p[i]] == t[i + p[i]]:
            p[i] += 1
        # 更新最右边界和中心
        if i + p[i] > mx:
            mx = i + p[i]
            id = i
        # 更新最长回文子串的信息
        if p[i] - 1 > max_len:
            max_len = p[i] - 1
            center_index = i
    # 计算最长回文子串的起始位置和结束位置
    start = (center_index - max_len) // 2
    end = start + max_len
    return s[start:end]
if __name__ == '__main__':
    s1 = "aba"
    s2 = "abba"
    s3 = "abccba"
    s4 = "abcddcba"
    s5 = "abbac"
    s6 = "a"
    s7 = "aa"
    s8 = "abc"
    print(f"'{s1}'的最长回文子串: {manacher(s1)}")  # 输出: aba
    print(f"'{s2}'的最长回文子串: {manacher(s2)}")  # 输出: abba
    print(f"'{s3}'的最长回文子串: {manacher(s3)}")  # 输出: abccba
    print(f"'{s4}'的最长回文子串: {manacher(s4)}")  # 输出: abcddcba
    print(f"'{s5}'的最长回文子串: {manacher(s5)}")  # 输出: abba
    print(f"'{s6}'的最长回文子串: {manacher(s6)}")  # 输出: a
    print(f"'{s7}'的最长回文子串: {manacher(s7)}")  # 输出: aa
    print(f"'{s8}'的最长回文子串: {manacher(s8)}")  # 输出: a
```

#### 最长回文子串

```python
s = input()
lth = len(s)
if lth == 1:
    print(s)
else:
    max_length = 1
    start = 0
    dp = [[False]*lth for _ in range(lth)]
    for i in range(lth):
        dp[i][i] = True
    for l in range(2,lth+1):
        for i in range(lth-l+1):
            j = i + l - 1
            if s[i] == s[j] and (j - i <= 2 or dp[i+1][j-1]):
                dp[i][j] = True
            if dp[i][j] and j - i + 1 > max_length:
                max_length = j - i + 1
                start = i
    print(s[start:start+max_length])
```

### DP动态规划

#### 1. **0-1 背包问题**
**状态转移方程**：  

- 如果不选第 \(i\) 件物品：`dp[i][j] = dp[i-1][j]`
- 如果选第 \(i\) 件物品：`dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]] + v[i])`

```python
def knapsack_01(weights, values, capacity):
    n = len(weights)
    # 初始化 DP 数组
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    # 遍历物品
    for i in range(1, n + 1):
        for j in range(capacity + 1):
            if j < weights[i - 1]:  # 当前容量不足以放下第 i 件物品
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i - 1]] + values[i - 1])
    return dp[n][capacity]
# 示例
weights = [1, 2, 3]
values = [6, 10, 12]
capacity = 5
print(knapsack_01(weights, values, capacity))  # 输出：22
```

**优化到一维数组**：

```python
def knapsack_01_optimized(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for j in range(capacity, weights[i] - 1, -1):  # 倒序遍历
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[capacity]
# 示例
print(knapsack_01_optimized(weights, values, capacity))  # 输出：22
```

---

#### 2. **完全背包问题**
**问题描述**：每种物品可以选择任意次，问如何选择物品使得总价值最大。

**状态转移方程**：  

- 如果不选第 \(i\) 件物品：`dp[i][j] = dp[i-1][j]`
- 如果选第 \(i\) 件物品：`dp[i][j] = max(dp[i][j], dp[i][j-w[i]] + v[i])`

```python
def knapsack_complete(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for j in range(weights[i], capacity + 1):  # 正序遍历
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[capacity]
# 示例
weights = [1, 2, 3]
values = [6, 10, 12]
capacity = 5
print(knapsack_complete(weights, values, capacity))  # 输出：30
```

#### 3. **多重背包问题**
**问题描述**： 每种物品有一个有限的选取次数 \(k[i]\)，问如何选择物品使得总价值最大。

**状态转移方程**： 枚举选取第 \(i\) 件物品的次数 \(t\)，`dp[j] = max(dp[j], dp[j - t * w[i]] + t * v[i])`

**优化为二进制拆分**：  将每种物品拆分为若干件，转化为 0-1 背包。

```python
def knapsack_multiple(weights, values, counts, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        if counts[i] * weights[i] >= capacity:  # 转化为完全背包
            for j in range(weights[i], capacity + 1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        else:  # 0-1 背包的二进制优化
            k = 1
            count = counts[i]
            while k < count:
                for j in range(capacity, k * weights[i] - 1, -1):
                    dp[j] = max(dp[j], dp[j - k * weights[i]] + k * values[i])
                count -= k
                k *= 2
            for j in range(capacity, count * weights[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - count * weights[i]] + count * values[i])
    return dp[capacity]
```

---

#### 4. **分组背包问题**
**问题描述**： 物品被分成若干组，每组最多选一件。问如何选择物品使得总价值最大。

```python
def knapsack_group(groups, capacity):
    dp = [0] * (capacity + 1)
    for group in groups:
        for j in range(capacity, -1, -1):  # 倒序遍历容量
            for weight, value in group:
                if j >= weight:
                    dp[j] = max(dp[j], dp[j - weight] + value)
    return dp[capacity]
```

---

#### 5. **混合背包问题**
```python
def knapsack_mixed(weights, values, counts, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        if counts[i] == 0:  # 完全背包
            for j in range(weights[i], capacity + 1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        elif counts[i] == 1:  # 0-1 背包
            for j in range(capacity, weights[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        else:  # 多重背包
            k = 1
            count = counts[i]
            while k < count:
                for j in range(capacity, k * weights[i] - 1, -1):
                    dp[j] = max(dp[j], dp[j - k * weights[i]] + k * values[i])
                count -= k
                k *= 2
            for j in range(capacity, count * weights[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - count * weights[i]] + count * values[i])
    return dp[capacity]
```

### 双指针

#### 接雨水

```python
heights = list(map(int, input().split()))
lth = len(heights)
result = left = left_max = right_max = 0
right = lth - 1
while left < right:
    left_max = max(left_max,heights[left])
    right_max = max(right_max, heights[right])
    if left_max < right_max:
        result += left_max - heights[left]
        left += 1
    else:
        result += right_max - heights[right]
        right -= 1
print(result)
```

### Kadane算法"最大子数组和"

1. 基础版本：
```python
def kadane(arr):
    max_sum = float('-inf')  # 全局最大和
    current_sum = 0          # 当前和
    for num in arr:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
        
    return max_sum
```

2. 同时返回子数组起始和结束位置：
```python
def kadane_with_position(arr):
    max_sum = float('-inf')  # 全局最大和
    current_sum = 0          # 当前和
    start = 0               # 开始位置
    end = 0                 # 结束位置
    temp_start = 0          # 临时开始位置
    for i, num in enumerate(arr):
        if current_sum <= 0:
            current_sum = num
            temp_start = i
        else:
            current_sum += num
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    return max_sum, start, end
```

3. 处理全负数数组的版本：
```python
def kadane_all_negative(arr):
    max_sum = max(arr)      # 如果全是负数，返回最大的负数
    current_sum = 0
    for num in arr:
        current_sum = max(0, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum
```

4. 循环数组的最大子数组和（允许首尾相连）：
```python
def kadane_circular(arr):
    # 普通kadane
    def kadane(arr):
        max_sum = float('-inf')
        current_sum = 0
        for num in arr:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        return max_sum
    # 1. 找到普通的最大子数组和
    max_normal = kadane(arr)
    # 2. 找到数组总和
    total = sum(arr)
    # 3. 将所有元素取反，再次使用kadane
    # 这样找到的是最小子数组和的相反数
    max_circular = total + kadane([-x for x in arr])
    # 如果数组全为负数，max_circular将为0
    if max_circular == 0:
        return max_normal
    return max(max_normal, max_circular)
```

#### 最大子矩阵

```python
def max_submatrix_dp(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')
    # 预处理：计算列的前缀和
    for r in range(1, rows):
        for c in range(cols):
            matrix[r][c] += matrix[r-1][c]
    for r1 in range(rows):
        for r2 in range(r1, rows):
            # 将每列的子矩阵和看作一个一维数组
            temp_arr = [0] * cols
            for c in range(cols):
                temp_arr[c] = matrix[r2][c] - (matrix[r1-1][c] if r1 > 0 else 0)
            # 使用 Kadane 算法求一维数组的最大子数组和
            current_max = 0
            for num in temp_arr:
                current_max = max(num, current_max + num)
                max_sum = max(max_sum, current_max)
    return max_sum
```

**方法三：优化动态规划**

```python
def max_submatrix_dp_optimized(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')
    # 预处理：计算列的前缀和
    for r in range(1, rows):
        for c in range(cols):
            matrix[r][c] += matrix[r-1][c]
    for r1 in range(rows):
        temp_arr = [0] * cols  # 复用 temp_arr
        for r2 in range(r1, rows):
            # 计算当前子矩阵的和
            for c in range(cols):
                temp_arr[c] = matrix[r2][c] - (matrix[r1-1][c] if r1 > 0 else 0)
            # 使用 Kadane 算法求一维数组的最大子数组和
            current_max = 0
            for num in temp_arr:
                current_max = max(num, current_max + num)
                max_sum = max(max_sum, current_max)
    return max_sum
```

### 最长上升子序列

**方法一：动态规划（DP）**使用一个 `dp` 数组，其中 `dp[i]` 表示以 `nums[i]` 结尾的最长上升子序列的长度。

```python
def longest_increasing_subsequence_dp(nums):
    n = len(nums)
    if n == 0:
        return 0
    dp = [1] * n  # 初始化 dp 数组
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

**方法二：贪心 + 二分查找**

```python
import bisect
def longest_increasing_subsequence_binary_search(nums):
    tails = []
    for num in nums:
        i = bisect.bisect_left(tails, num)  # 二分查找
        if i == len(tails):
            tails.append(num)
        else:
            tails[i] = num
    return len(tails)
```

### 滑雪（记忆化搜索）

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

##### 矩阵路径问题

```python
def unique_paths_memo(m, n, memo={}):
    if (m, n) in memo:
        return memo[(m, n)]
    if m == 1 or n == 1:
        return 1
    result = unique_paths_memo(m - 1, n, memo) + unique_paths_memo(m, n - 1, memo)
    memo[(m, n)] = result
    return result
```

