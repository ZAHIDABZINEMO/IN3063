# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 10:26:49 2022

@author: zahid
"""

import random
from heapq import heappush, heappop

class Grid:
    def __init__(self, height, width):
        """
        Initializes a grid with the given height and width.
        The grid is initialized with all values set to 0, and the start position is set to (0, 0) and the goal position is set to (height - 1, width - 1).
        """
        self.height = height
        self.width = width
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.start = (0, 0)
        self.goal = (height - 1, width - 1)
    
    def generate_grid(self, distribution):
        """
        Generates random values for the cells in the grid from 0 to 9 using the given distribution.
        """
        for row in self.grid:
            for i in range(len(row)):
                row[i] = distribution()
    
    def display(self):
        """
        Prints the grid to the console.
        """
        for row in self.grid:
            print(row)

def random_distribution():
    """
    Returns a random integer from 0 to 9 (inclusive).
    """
    return random.randint(0, 9)

"""
Get height and width from user input and create a grid with randomly generated numbers
"""
height = int(input("Enter the height of the grid: "))
width = int(input("Enter the width of the grid: "))

g = Grid(height, width)
g.generate_grid(random_distribution)
g.display()

"""
Use of greedy search for Heuristic algorithm to find short paths. 
"""
def greedy_search(grid):
    """
    Conducts a greedy search on the given grid to find the shortest path to the goal.
    At each step, the agent will choose the adjacent cell that brings it closer to the goal based on the Manhattan distance (the sum of the absolute differences of the coordinates).
    Returns a list of tuples representing the coordinates of the cells visited in the path.
    """
    path = []
    current_pos = grid.start
    while current_pos != grid.goal:
        path.append(current_pos)
        min_distance = float('inf')
        next_pos = None
        for r, c in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # Check if the new position is within the grid bounds
            row = current_pos[0] + r
            col = current_pos[1] + c
            if row >= 0 and row < grid.height and col >= 0 and col < grid.width:
                # Calculate the Manhattan distance from the new position to the goal
                distance = abs(row - grid.goal[0]) + abs(col - grid.goal[1])
                if distance < min_distance:
                    # Update the minimum distance and the next position if a shorter distance is found
                    min_distance = distance
                    next_pos = (row, col)
        current_pos = next_pos
    path.append(grid.goal)
    return path

# Example usage
path = greedy_search(g)
print(path)

def dijkstra(grid, start, goal):
    """
    Conducts a Dijkstra search on the given grid to find the shortest path from the start position to the goal position.
    The grid is assumed to have a uniform cost for each cell.
    Returns a dictionary with the keys being the coordinates of the cells and the values being the distance from the start position.
    """
    # Initialize distances with infinity
    distances = {(row, col): float('inf') for row in range(grid.height) for col in range(grid.width)}
    # Set distance of starting node to 0
    distances[start] = 0
    
    # Initialize priority queue with starting node
    pq = [(0, start)]
    
    while pq:
        # Get node with smallest distance from priority queue
        distance, current_pos = heappop(pq)
        if distance > distances[current_pos]:
            # Skip nodes that have been visited with a shorter distance
            continue
        # Update distances of adjacent nodes
        for r, c in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            row = current_pos[0] + r
            col = current_pos[1] + c
            if row >= 0 and row < grid.height and col >= 0 and col < grid.width:
                next_pos = (row, col)
                cost = grid.grid[row][col]
                if distances[current_pos] + cost < distances[next_pos]:
                    distances[next_pos] = distances[current_pos] + cost
                    # Add adjacent node to priority queue with updated distance
                    heappush(pq, (distances[next_pos], next_pos))
    return distances

# Example usage
g = Grid(height, width)
g.generate_grid(random_distribution)
distances = dijkstra(g, g.start, g.goal)
path_length = distances[g.goal]
print(path_length)
