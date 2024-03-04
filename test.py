"""
Algorithm from Optimal Output-Sensitive Convex Hull Algorithms in Two and Three Dimensions by Timothy M. Chan (1996)
Algorithm Hull2D(P, m, H), where P C E^2, 3 <= m <= n, and H >= 1
1. partition P into subsets P_1..... P_[n/m] each of size at most m
2. for i = 1,..., [n/m] do
3.      compute conv(Pi) by Graham's scan and store its vertices in an array
        in ccw order
4. P0 <- (0,-float('inf'))
5. P1 <- the rightmost point of P
6. for k = 1,...,H do
7.      for i = 1,...,[n/m] do
8.              compute the point q_i in P_i that maximizes angle between p_k-1, P_k and q_i (q_i =/= P_k)
                by performing a binary search on the vertices of conv(Pi)
9.      p_k+1 <- the point q from {q_1,.....,q_[n/m]}that maximizes angle between p_(k-1) and p_k q
10.     if p_k+1 = p_i then return the list (p_1..... p_k)
11. return incomplete
"""

import math
# Graham Scan Algorithm is taken from Chapter 33 of Introduction to Algorithms (3rd Edition) by Cormen et al. (2009). Python implementation is based on the pseudocode provided in the book.

# function to get the polar angle between two points in 2D space - to determine the angle between the x-axis and the line segment connecting the two points, used for sorting the points in the graham scan algorithm.
def getPolarAngle(v1 : tuple[int, int], v2 : tuple[int, int]) -> float:
# calculate the polar angle
        polarAngle = math.atan2(v2[1] - v1[1], v2[0] - v1[0])
        
        # if the polar angle is negative, then add 2pi to the angle to make it positive, so it is in the range of 0 to 2pi (following polar coordinates convention).
        if polarAngle < 0:
                polarAngle += (2 * math.pi)
        
        return polarAngle

# function to get the distance between two points in 2D space.
def getDistance(v1 : tuple[int,int],v2 : tuple[int,int]) -> float:
    return math.sqrt((v2[0]-v1[0])**2 + (v2[1]-v1[1])**2)

# cross product is used to determine whether consctuive segments turn left or right or are collinear (straight).
def crossProduct(v1 : tuple[int,int], v2 : tuple[int,int], v3 : tuple[int,int]) -> float:
    # formula for cross product:
    # (p1 - p0) x (p2 - p0) = (x1 - x0)(y2 - y0) - (y1 - y0)(x2 - x0)
    # positive crossproduct means left turn at p1 (counterclockwise), negative value means right turn at p2 (clockwise), and 0 means collinear
    return (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])

# function to compare the points based on the polar angle and distance from p0 (for the other 3 sorting algorithms)
def compare_points(p0 : tuple[int,int], p1 : tuple[int,int], p2 : tuple[int,int]) -> float:
    angle1 = getPolarAngle(p0, p1)
    distance1 = getDistance(p0, p1)
    angle2 = getPolarAngle(p0, p2)
    distance2 = getDistance(p0, p2)
    if angle1 == angle2:
        return distance1 - distance2
    else:
        return angle1 - angle2

# Merge Sort
def mergeSort(inputSet : list[tuple[int,int]], compare, start=0, end=None) -> list[tuple[int,int]]:
    if end is None:
        end = len(inputSet)
    if end - start <= 1:
        return
    mid = start + (end - start) // 2
    mergeSort(inputSet, compare, start, mid)
    mergeSort(inputSet, compare, mid, end)
    merge(inputSet, compare, start, mid, end)

def merge(inputSet : list[tuple[int,int]], compare, start, mid, end) -> list[tuple[int,int]]:
    left = inputSet[start:mid]
    right = inputSet[mid:end]
    i = j = 0
    for k in range(start, end):
        if i < len(left) and (j >= len(right) or compare(left[i], right[j]) <= 0):
            inputSet[k] = left[i]
            i += 1
        else:
            inputSet[k] = right[j]
            j += 1

# graham scan algorithm with python inbuilt sort function (timsort - O(nlogn) time complexity) (average case) [main function]!
def grahamscan(inputSet : list[tuple[int,int]]) -> list[tuple[int,int]]:
    '''
    Returns the list of points that lie on the convex hull (graham scan algorithm)
            Parameters:
                    inputSet (list): a list of 2D points

            Returns:
                    outputSet (list): a list of 2D points
    '''
    # if the inputSet has less than 3 points, then return the inputSet as the convex hull
    if len(inputSet) < 3:
        return inputSet
    # find the point with the minimum y-coordinate or leftmost point in case of a tie
    p0 = min(inputSet, key=lambda x:(x[1], x[0]))

    """
    let p1, p2,...,pi be the remaining points in Q,
    sorted by polar angle in counterclockwise order around p0
    (if more than one point has the same angle, remove all but
    the one that is farthest from p0)
    """

    # Sort the points based on the polar angle, and if the polar angle is the same, then sort based on the furthest distance from p0 in the case of a tie, with a lambda function.
    inputSet = sorted(inputSet, key=lambda x: (getPolarAngle(p0, x), getDistance(p0, x)))

    # initialize a stack, outputSet, to store the vertices of the convex hull
    outputSet = [p0, inputSet[1], inputSet[2]]

    for i in range(3, len(inputSet)):
        # while the angle formed by points next-to-top(s), top(s) and p_i makes a nonleft turn, pop from stack
        while len(outputSet) > 1 and crossProduct(outputSet[-2], outputSet[-1], inputSet[i]) <= 0:
            outputSet.pop()
        outputSet.append(inputSet[i])
    return outputSet

def grahamscanMergeSort(inputSet : list[tuple[int,int]]) -> list[tuple[int,int]]:
    if len(inputSet) < 3:
        return inputSet
    p0 = min(inputSet, key=lambda x:(x[1], x[0]))
    mergeSort(inputSet, lambda x, y: compare_points(p0, x, y))
    outputSet = [p0, inputSet[1], inputSet[2]]
    for i in range(3, len(inputSet)):
        # while the angle formed by points next-to-top(s), top(s) and p_i makes a nonleft turn, pop from stack
        while len(outputSet) > 1 and crossProduct(outputSet[-2], outputSet[-1], inputSet[i]) <= 0:
            outputSet.pop()
        outputSet.append(inputSet[i])
    return outputSet

# Insertion Sort
def insertionSort(inputSet : list[tuple[int,int]], compare) -> list[tuple[int,int]]:
    for i in range(1, len(inputSet)):
        key = inputSet[i]
        j = i - 1
        while j >= 0 and compare(inputSet[j], key) > 0:
            inputSet[j + 1] = inputSet[j]
            j -= 1
        inputSet[j + 1] = key
    return inputSet

# graham scan with insertion sort
def grahamscanInsertionSort(inputSet : list[tuple[int,int]]) -> list[tuple[int,int]]:
    if len(inputSet) < 3:
        return inputSet
    p0 = min(inputSet, key=lambda x:(x[1], x[0]))
    inputSet = insertionSort(inputSet, lambda x, y: compare_points(p0, x, y))
    outputSet = [p0, inputSet[1], inputSet[2]]
    for i in range(3, len(inputSet)):
        # while the angle formed by points next-to-top(s), top(s) and p_i makes a nonleft turn, pop from stack
        while len(outputSet) > 1 and crossProduct(outputSet[-2], outputSet[-1], inputSet[i]) <= 0:
            outputSet.pop()
        outputSet.append(inputSet[i])
    return outputSet



# https://en.wikipedia.org/wiki/Chan%27s_algorithm

# cross product is used to determine whether consctuive segments turn left or right or are collinear (straight).
def crossProduct(v1 : tuple[int,int], v2 : tuple[int,int], v3 : tuple[int,int]) -> int:
    # formula for cross product:
    # (p1 - p0) x (p2 - p0) = (x1 - x0)(y2 - y0) - (y1 - y0)(x2 - x0)
    # positive crossproduct means left turn at p2 (counterclockwise), negative value means right turn at p2 (clockwise), and 0 means collinear
    value = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])
    if value > 0:
            # left turn
        return 1
    elif value < 0:
            # right turn
        return -1
    else:
            # collinear
        return 0

# find distance between two points on the 2D plane.
def dist(p1 : tuple[int,int], p2: tuple[int,int]) -> float:
        return math.sqrt((p2[1]-p1[1])**2 +(p2[0]-p1[0])**2)

# find the point that has the rightmost tangent from current point in one of the subhulls via jarvis binary search on one subhull. - taken from https://gist.github.com/tixxit/252229#file-chan-py-L45
def jarvisBinarySearch(hull : list[tuple[int,int]], p : tuple[int,int]) -> int:
        left = 0
        right = len(hull)
        middle = 0
        left_prev = crossProduct(p, hull[0], hull[-1])
        left_next = crossProduct(p, hull[0], hull[(left+1) % right])
        while left < right:
                middle = (right + left) // 2
                middle_prev = crossProduct(p, hull[middle], hull[(middle-1) % len(hull)])
                middle_next = crossProduct(p, hull[middle], hull[(middle+1) % len(hull)])
                middle_side = crossProduct(p, hull[left], hull[middle])
                # check if the previous and next turns are not clockwise, then return the middle index as the rightmost tangent.
                if middle_prev != -1 and middle_next != -1:
                        #print("middle before return:", middle)
                        return hull[middle]
                # otherwise if the point pointed by middle is counter-clockwise but the point pointed by middle_next is clockwise, then the tangent touches the right chain.
                elif middle_side == 1 and (left_next == -1 or left_prev == left_next) or \
                        middle_side == -1 and middle_prev == -1:
                        # tangent touches the left chain
                        right = middle
                # otherwise tangent touches the right chain - reduce the left index to middle + 1
                else:
                        left = middle + 1
                        left_prev = -middle_next
                        
                        if left < len(hull):
                                left_next = crossProduct(p, hull[left], hull[(left+1) % (len(hull))])
                        else:
                                return hull[-1]
        return hull[left]

# pick the starting point of the convex hull based on the rightmost point
def pick_start(hulls : list[list[tuple[int,int]]]) -> tuple[int,int]:
    rightmost_point = max(hulls, key=lambda x: (x[0],-x[1]))
    return rightmost_point

# find the index of the next point in the convex hull based on the rightmost tangent - from the current point in the convex hull 
def next_point_pair(hulls : list[list[tuple[int,int]]], pair : tuple[int,int]) -> tuple[int,int]:
        # obtain the current coordinate passed based on the indexes passed in the last element of outputSet.
        current_point = pair
        indices = [(i, sublist.index(current_point)) for i, sublist in enumerate(hulls) if current_point in sublist]
        next = (indices[0][0], (indices[0][1] + 1) % len(hulls[indices[0][0]]))
        next_coord = hulls[next[0]][next[1]]
        for i in range(len(hulls)):
                if i == next[0]:
                        continue
                # use binary search to get the coordinates with the rightmost tangent in the next subhull
                next_point = jarvisBinarySearch(hulls[i], current_point)
                print(next)
                
                # get the coordinate of the next point in the current subhull
                q = hulls[next[0]][next[1]]
                
                
                # check the turn of the point
                t = crossProduct(current_point, q, next_point)
                if t == -1 or (t == 0 and dist(current_point, next_point) > dist(current_point, q)):
                        next_coord = next_point
        return next_coord

# Chan's Algorithm with Binary Search:
def chen(inputSet : list[tuple[int,int]]) -> list[tuple[int,int]]:
        '''
        Returns the list of points that lie on the convex hull (chen's algorithm)
                Parameters:
                        inputSet (list): a list of 2D points
        
                Returns:
                        outputSet (list): a list of 2D points
        '''
        # Let m = 4, so there is no need for extra iterations to make the algorithm slower.
        m = 3
        if len(inputSet) < 3:
                return inputSet
        
        
        
        # note that h <= m <= h^2 for all h >= 1 in the algorithm - so we do not perform too many iterations - based on Wikipedia. 
        h = min(2 ** (2**m), len(inputSet))
        
        p1 = pick_start(inputSet)
        outputSet = [p1]
        
        """Partition the inputSet into subsets of size [n/m] with at most m using list slicing and list comprehension, 
        and then compute the convex hull of each subset using Graham's scan 
        - store the vertices in an array in counterclockwise order."""
        subhulls = [grahamscan(inputSet[i:i+h]) for i in range(0, len(inputSet), h)]
        # Using a modified Jarvis March with binary search to obtain the convex hull points and merge the subhulls to form the convex hull.
        for _ in range(len(inputSet)):
                p = next_point_pair(subhulls, outputSet[-1])
                if p == outputSet[0]:
                        return outputSet
                outputSet.append(p)
        return None

        """
        #Chan's Algorithm with Linear Search (based on a jarvis march on graham scan of hulls):
        
        h = min(2 ** (2**m), len(inputSet))

        # Partition the inputset into subsets of size at most m using list slicing and list comprehension.
        subsets = [inputSet[i:i+h] for i in range(0, len(inputSet), h)]
        # Compute the convex hull of each subset using Graham's scan - store the vertices in an array in counterclockwise order
        subhulls = [grahamscan(subset) for subset in subsets]

        # Merge the subhulls using Jarvis's march
        outputSet = jarvismarch([point for subhull in subhulls for point in subhull])
        
        # check if the outputSet is less than or equal to h, then return the outputSet
        if len(outputSet) <= h:
                return outputSet 
        
        return None
        """





hull =[(17385, 0), (25978, 0), (29111, 4), (30672, 10), (32048, 23), (32426, 167), (32672, 431), (32754, 688), (32767, 2407), (32765, 17830), (32762, 28766), (32759, 31071), (32744, 32563), (32719, 32652), (32226, 32752), (26862, 32761), (16519, 32766), (3668, 32757), (284, 32698), (90, 32480), (28, 32138), (16, 31084), (14, 30640), (7, 21680), (1, 2654), (15, 1382), (69, 248), (109, 186), (221, 108), (1123, 15), (1977, 6), (2639, 5)]
x = chen(hull)
print(x)
