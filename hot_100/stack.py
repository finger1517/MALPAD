from typing import List, Optional

def isValid(s: str) -> bool:
    """
    有效的括号
    
    问题：判断字符串中的括号是否有效（匹配且顺序正确）
    
    思路：
    1. 使用栈来匹配括号
    2. 遇到左括号时入栈
    3. 遇到右括号时，检查栈顶是否匹配的左括号
    4. 如果匹配，弹出栈顶；如果不匹配，返回 False
    5. 最后检查栈是否为空
    
    栈的应用：
    - 后进先出（LIFO）特性完美匹配括号的嵌套结构
    - 时间复杂度 O(n)，空间复杂度 O(n)
    
    示例：s = "()[]{}"
    - '(': 入栈 -> stack = ['(']
    - ')': 匹配，弹出 -> stack = []
    - '[': 入栈 -> stack = ['[']
    - ']': 匹配，弹出 -> stack = []
    - '{': 入栈 -> stack = ['{']
    - '}': 匹配，弹出 -> stack = []
    - 返回 True
    
    时间复杂度：O(n)，空间复杂度：O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    return not stack

class MinStack:
    """
    最小栈
    
    功能：
    - 支持标准的栈操作（push, pop, top）
    - 能够在常数时间内获取栈中的最小元素
    
    思路：
    1. 使用两个栈：
       - 主栈：存储所有元素
       - 最小栈：存储每个位置对应的最小值
    2. push 操作：
       - 主栈直接入栈
       - 最小栈入栈当前元素和栈顶最小值的较小值
    3. pop 操作：两个栈同时弹出
    4. getMin 操作：直接返回最小栈的栈顶
    
    优化：
    - 可以只在必要时更新最小栈，节省空间
    - 但这样会增加 pop 操作的复杂度
    
    示例：
    - push(-2): stack=[-2], min_stack=[-2]
    - push(0): stack=[-2,0], min_stack=[-2,-2]
    - push(-3): stack=[-2,0,-3], min_stack=[-2,-2,-3]
    - getMin(): -3
    - pop(): stack=[-2,0], min_stack=[-2,-2]
    - top(): 0
    - getMin(): -2
    
    时间复杂度：
    - push: O(1)
    - pop: O(1)
    - top: O(1)
    - getMin: O(1)
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
        else:
            self.min_stack.append(self.min_stack[-1])
    
    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()
    
    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1]

def decodeString(s: str) -> str:
    """
    字符串解码
    
    问题：解码编码后的字符串，格式为 k[encoded_string]
    
    思路：
    1. 使用栈来处理嵌套结构
    2. 遇到数字时，累积计算数字（可能有多位）
    3. 遇到 '[' 时，将当前字符串和数字入栈
    4. 遇到 ']' 时，弹出数字和之前的字符串，进行解码
    5. 遇到字母时，直接拼接到当前字符串
    
    栈的应用：
    - 处理嵌套结构
    - 保存上下文信息（之前的字符串和重复次数）
    
    示例：s = "3[a2[c]]"
    - curr_num=0, curr_str=''
    - '3': curr_num=3
    - '[': stack=[], stack=[3], curr_str='', curr_num=0
    - 'a': curr_str='a'
    - '2': curr_num=2
    - '[': stack=['',3], stack=['',3,2], curr_str='', curr_num=0
    - 'c': curr_str='c'
    - ']': num=2, prev_str='', curr_str='' + 'c'*2 = 'cc'
    - ']': num=3, prev_str='', curr_str='' + 'a' + 'cc'*3 = 'accaccacc'
    
    时间复杂度：O(n)，空间复杂度：O(n)
    """
    stack = []
    curr_num = 0
    curr_str = ''
    for char in s:
        if char.isdigit():
            curr_num = curr_num * 10 + int(char)
        elif char == '[':
            stack.append(curr_str)
            stack.append(curr_num)
            curr_str = ''
            curr_num = 0
        elif char == ']':
            num = stack.pop()
            prev_str = stack.pop()
            curr_str = prev_str + curr_str * num
        else:
            curr_str += char
    return curr_str

def dailyTemperatures(temperatures: List[int]) -> List[int]:
    """
    每日温度
    
    问题：对于每一天，计算需要等待多少天才能遇到更高的温度
    
    思路：
    1. 使用单调栈来找到下一个更高的温度
    2. 栈中存储的是温度的索引，保持栈中对应的温度单调递减
    3. 对于每个温度，如果比栈顶温度高，则可以更新栈顶位置的等待天数
    4. 栈中的索引对应的温度都是递减的，这样可以保证正确计算等待天数
    
    单调栈：
    - 递减栈：找到下一个更大的元素
    - 递增栈：找到下一个更小的元素
    
    示例：temperatures = [73,74,75,71,69,72,76,73]
    - i=0: stack=[0]
    - i=1: 74>73 -> result[0]=1, stack=[1]
    - i=2: 75>74 -> result[1]=1, stack=[2]
    - i=3: 71<75 -> stack=[2,3]
    - i=4: 69<71 -> stack=[2,3,4]
    - i=5: 72>69 -> result[4]=1, stack=[2,3,5]
    - i=5: 72>71 -> result[3]=2, stack=[2,5]
    - i=6: 76>72 -> result[5]=1, stack=[2,6]
    - i=6: 76>75 -> result[2]=4, stack=[6]
    - i=7: 73<76 -> stack=[6,7]
    - result = [1,1,4,2,1,1,0,0]
    
    时间复杂度：O(n)，空间复杂度：O(n)
    """
    n = len(temperatures)
    result = [0] * n
    stack = []
    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)
    return result

def largestRectangleArea(heights: List[int]) -> int:
    """
    柱状图中最大的矩形
    
    问题：在柱状图中找到最大的矩形面积
    
    思路：
    1. 使用单调栈来找到每个柱子的左右边界
    2. 左边界：左边第一个比当前柱子矮的位置
    3. 右边界：右边第一个比当前柱子矮的位置
    4. 矩形面积 = height × (right - left - 1)
    5. 使用技巧：在数组末尾添加 0，确保所有柱子都能被处理
    
    单调栈：
    - 递增栈：用于寻找左右边界
    - 当遇到比栈顶小的元素时，可以计算栈顶元素的矩形面积
    
    示例：heights = [2,1,5,6,2,3]
    - 添加 0：heights = [2,1,5,6,2,3,0]
    - i=0: stack=[0]
    - i=1: 1<2 -> height=2, width=1, area=2, stack=[1]
    - i=2: 5>1 -> stack=[1,2]
    - i=3: 6>5 -> stack=[1,2,3]
    - i=4: 2<6 -> height=6, width=3-2-1=1, area=6, stack=[1,2,4]
    - i=4: 2<5 -> height=5, width=4-1-1=2, area=10, stack=[1,4]
    - i=5: 3>2 -> stack=[1,4,5]
    - i=6: 0<3 -> height=3, width=6-4-1=1, area=3, stack=[1,4]
    - i=6: 0<2 -> height=2, width=6-1-1=4, area=8, stack=[1]
    - i=6: 0<1 -> height=1, width=6, area=6, stack=[6]
    - 最大面积：10
    
    时间复杂度：O(n)，空间复杂度：O(n)
    """
    stack = []
    max_area = 0
    heights.append(0)
    for i in range(len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    heights.pop()
    return max_area