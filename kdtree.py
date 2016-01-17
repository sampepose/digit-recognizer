# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 21:24:48 2016

@author: sampepose
"""

import numpy as np
import operator

class Node:
    def __init__(self, location, value):
        self.location = location
        self.value = value
        self.left = None
        self.right = None
             
class BoundedPriorityQueue:
    def __init__(self, capacity):   
        self.list = []
        self.size = 0
        self.capacity = capacity
        
    def push(self, item, priority):
        if self.full() is True:
            if priority < self.peek()[1]: # less than because distances are kinda reverse priority
                self.list.pop()
                self.size -= 1
            else:
                return
                        
        self.list.append([item, priority])
        self.list.sort(key=operator.itemgetter(1))
        self.size += 1            
            
    # returns the top element (lowest priority)
    def peek(self):
        if self.size == 0:
            return None
        else:
            return self.list[0]
                        
    def full(self):
        return self.size >= self.capacity
        
    def maxPriority(self):
        if self.size == 0:
            return None
        else:
            return self.list[self.size - 1][1]
        
class KdTree:
    def __init__(self, locations, values):   
        self.k = locations.shape[1]
        self.root = self.__construct(locations, values, 0)
        
    def __construct(self, locations, values, depth):
        if locations.size == 0:
            return None
        
        # index of the column to sort by        
        axis = depth % self.k
        
        # sort data by (depth % k)th index
        sorted_idxs = locations[:, axis].argsort()
        sorted_data = locations[sorted_idxs]
        sorted_vals = values[sorted_idxs]
        
        # grab the median location and its value
        median_idx = sorted_data.shape[0] // 2
        median_loc = sorted_data[median_idx, :]
        median_val = sorted_vals[median_idx]
        
        # find the largest index of points with median value
        while median_idx + 1 < sorted_data.shape[0] and \
            sorted_data[median_idx + 1, axis] == median_loc[axis]:
            median_idx += 1
        
        # create a new node and recursively solve for left and right nodes
        node = Node(median_loc, median_val)
        node.left = self.__construct(sorted_data[:median_idx,:], sorted_vals[:median_idx], depth+1)        
        node.right = self.__construct(sorted_data[median_idx+1:,:], sorted_vals[median_idx+1:], depth+1)        

        return node

    def __distance(self, nodeA, nodeB):
        return np.sum(np.abs(nodeA - nodeB))
        
    def kNN(self, point, k):
        topK = BoundedPriorityQueue(k)
        self.__kNN_helper(point, self.root, topK)
        return topK.list
            
    def __kNN_helper(self, point, node, topK, depth=0):
        if node is None:
            return
                        
        axis = depth % self.k
        
        topK.push(node, self.__distance(point, node.location))
        
        left = False
        if point[axis] <= node.location[axis]:
            left = True
            self.__kNN_helper(point, node.left, topK, depth+1)
        else:
            self.__kNN_helper(point, node.right, topK, depth+1)

        if not(topK.full()) or abs(node.location[axis] - point[axis]) < topK.maxPriority():
            if left is True: 
                self.__kNN_helper(point, node.right, topK, depth+1)
            else:
                self.__kNN_helper(point, node.left, topK, depth+1)