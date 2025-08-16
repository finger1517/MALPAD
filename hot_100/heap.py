import heapq

def findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]

def topKFrequent(nums, k):
    freq_map = {}
    for num in nums:
        freq_map[num] = freq_map.get(num, 0) + 1
    heap = []
    for num, freq in freq_map.items():
        heapq.heappush(heap, (-freq, num))
    result = []
    for _ in range(k):
        if heap:
            result.append(heapq.heappop(heap)[1])
    return result

class MedianFinder:
    def __init__(self):
        self.max_heap = []
        self.min_heap = []
    
    def addNum(self, num):
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def findMedian(self):
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return -self.max_heap[0]