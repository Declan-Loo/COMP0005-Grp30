import matplotlib.pyplot as plt
import timeit
import random
import math


def orientation(p, q, r):
    # Determine the orientation of the triplet (p, q, r).
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    elif val > 0:
        return 1  # Clockwise
    else:
        return 2  # Counterclockwise


def grahamscan1(points):
    # Handle cases where there are fewer than three points
    if len(points) < 3:
        # If there are less than three points, return them as they can't form a convex hull
        return points
    # Find the bottom-most point
    ymin = points[0][1]
    minIndex = 0
    for i, p in enumerate(points):
        if p[1] < ymin or (p[1] == ymin and p[0] < points[minIndex][0]):
            ymin = p[1]
            minIndex = i

    # Place the bottom-most point at the first position
    points[0], points[minIndex] = points[minIndex], points[0]
    referencePoint = points[0]

    # Sort the points based on their polar angle with respect to the reference point
    sortedPoints = sorted(points[1:], key=lambda p:
    (math.atan2(p[1] - referencePoint[1], p[0] - referencePoint[0]), p[0], p[1]))
    sortedPoints.insert(0, referencePoint)  # Add the reference point back to the start

    # Construct the convex hull
    hull = [sortedPoints[0], sortedPoints[1]]  # Start with the first two points

    for p in sortedPoints[2:]:
        while len(hull) >= 2 and orientation(hull[-2], hull[-1], p) != 2:
            hull.pop()  # Remove the last point from the hull if it's not a left turn
        hull.append(p)

    return hull

import math

def grahamscan2(inputSet):
    """
    Returns the list of points that lie on the convex hull (graham scan algorithm)
            Parameters:
                    inputSet (list): a list of 2D points

            Returns:
                    outputSet (list): a list of 2D points
    """

    def orientation(p1, p2, p3):
        # = 0 <- collinear
        # > 0 <- clockwise
        # < 0 <- counterclockwise
        return (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])

    def distanceSquared(p1, p2):
        return ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)

    n = len(inputSet)

    # Find the lowest coordinate
    ymin = inputSet[0][1]
    anchor = 0
    for i in range(1, n):
        y = inputSet[i][1]
        if y < ymin or (ymin == y and inputSet[i][0] < inputSet[anchor][0]):
            ymin = inputSet[i][1]
            anchor = i

    # Put the lowest coordinate at the beginning of the set
    inputSet[0], inputSet[anchor] = inputSet[anchor], inputSet[0]

    # Sort by polar angle
    lowestPoint = inputSet[0]

    inputSet.sort(key = lambda p: (math.atan2(p[1] - lowestPoint[1], p[0] - lowestPoint[0]), distanceSquared(lowestPoint, p)))

    # for i in range(1, n):
    #     j = i
    #     while j > 0 and (math.atan2(inputSet[j][1] - lowestPoint[1], inputSet[j][0] - lowestPoint[0]), distanceSquared(lowestPoint, inputSet[j])) < (math.atan2(inputSet[j - 1][1] - lowestPoint[1], inputSet[j - 1][0] - lowestPoint[0]), distanceSquared(lowestPoint, inputSet[j - 1])):
    #         inputSet[j], inputSet[j - 1] = inputSet[j - 1], inputSet[j]
    #         j -= 1

    # Deal with duplicates
    m = 1
    for i in range(1, n):
        while (i < n - 1) and (orientation(lowestPoint, inputSet[i], inputSet[i + 1]) == 0):
            i += 1
        inputSet[m] = inputSet[i]
        m += 1

    # # Convex hull needs at least 3 unique points
    # if m < 3:
    #     print("A convex hull needs at least 3 unique points")
    #     return

    S = []
    for i in range(3):
        S.append(inputSet[i])

    # Find the points on the convex hull
    for i in range(3, m):
        while len(S) > 1 and orientation(S[-2], S[-1], inputSet[i]) >= 0:
            S.pop()
        S.append(inputSet[i])

    outputSet = []
    while S:
        p = S[-1]
        outputSet.append(S.pop())
    return outputSet

def divide_points(inputSet : list[tuple[int,int]], m : int) -> list[list[tuple[int,int]]]:
        subsets = [inputSet[i:i+m] for i in range(0, len(inputSet), m)]
        return subsets

def chen(inputSet):
        '''
        Returns the list of points that lie on the convex hull (chen's algorithm)
                Parameters:
                        inputSet (list): a list of 2D points
        
                Returns:
                        outputSet (list): a list of 2D points
        '''
        # let m = 3
        if len(inputSet) < 3:
                return inputSet
        m = 3
        
        for t in range(m, len(inputSet)):
            
            # note that h <= m <= h^2 for all h >= 1 in the algorithm - so we do not perform too many iterations - based on Wikipedia.
            h = 2 ** (2 ** t)
            
            if h > len(inputSet):
                return grahamscan(inputSet)
            
            # Partition the inputset into subsets of size at most m using the divide_points function
            subsets = divide_points(inputSet, h)
            # Compute the convex hull of each subset using Graham's scan - store the vertices in an array in counterclockwise order
            subhulls = [grahamscan(subset) for subset in subsets]

            # Merge the subhulls using Jarvis's march
            outputSet = jarvismarch([point for subhull in subhulls for point in subhull])

            if len(outputSet) <= h:
                return outputSet
            
        return None

class TestDataGenerator():
    
    
    def __init__(self,num_points,radius=16383):
        self.num_points = num_points
        self.radius = radius
        

    def random_points(self):
        rand_points = []
        for _ in range(self.num_points):
            x = random.randint(0,32767)
            y = random.randint(0,32767)
            rand_points.append((x,y))
        return rand_points
    
    def generate_polygon_points(self,polygon_sides):
        polygon_vertices= []
        if polygon_sides <3:
            raise ValueError("A polygon must have at least 3 sides.")


        for i in range(polygon_sides):
            angle = 2 * math.pi * i / polygon_sides
            x = int(round(16383 + math.cos(angle) * self.radius))
            y = int(round(16383 + math.sin(angle) * self.radius))
            polygon_vertices.append((x, y))

        edge_points = []
        for i in range (polygon_sides):
            start_x, start_y = polygon_vertices[i]
            end_x, end_y = polygon_vertices[(i + 1) % polygon_sides]
            num_of_edge_points = math.ceil(self.num_points / polygon_sides)
            for j in range(num_of_edge_points):
                t = j/num_of_edge_points
                edge_x = start_x + t * (end_x - start_x)
                edge_y = start_y + t * (end_y - start_y)
                edge_points.append((edge_x, edge_y))
        

        return edge_points
    
    def generate_circle_points(self):
        circle_points = []
        centre_x = 0
        centre_y = 0

        for i in range(self.num_points):
            angle = 2 * math.pi * i / self.num_points
            x = int(round(centre_x + self.radius * math.cos(angle)))
            y = int(round(centre_y + self.radius * math.sin(angle)))
            circle_points.append((x, y))

        return circle_points
    
    
    def generate_collinear_points(self):
        grad = random.randint(-5,10)
        rand_points = []
        for i in range(self.num_points):
            x = i
            y = grad * x
            rand_points.append((x,y))
        return rand_points
    
def time_algorithm(algorithm, points):
    start = timeit.default_timer()
    algorithm(points)
    end = timeit.default_timer()
    return end - start

jarvis_times = []
graham_times = []
chan_times = []

    
def get_times(point_range):
    for i in point_range:
        data_generator = TestDataGenerator(i)
        points = data_generator.generate_circle_points()
        jarvis_times.append(time_algorithm(jarvismarch, points))
        graham_times.append(time_algorithm(grahamscan, points))
        chan_times.append(time_algorithm(chen, points))
        
def trials(point_range, n):
    jarvis_trials = []
    graham_trials = []
    chan_trials = []
    for j in point_range:
        jarvis_time = 0
        graham_time = 0
        chan_time = 0
        for i in range(n):
            data_generator = TestDataGenerator(j)
            points = data_generator.random_points()
            jarvis_time += time_algorithm(grahamscan1, points)
            graham_time += time_algorithm(grahamscan2, points)
            #chan_time += time_algorithm(chen, points)
        jarvis_trials.append(jarvis_time/n)
        graham_trials.append(graham_time/n)
        #chan_trials.append(chan_time/n)
        
    return jarvis_trials, graham_trials
        
    
point_range = [i for i in range(10,10000,100)]
jarvis_trials, graham_trials = trials(point_range, 20)

plt.plot(point_range, jarvis_trials, label = "Graham Scan 1", color = 'red')
plt.plot(point_range, graham_trials, label = "Graham Scan 2", color = 'blue')
#plt.plot(point_range, chan_trials, label = "Chan's Algorithm" , color = 'green')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(20.5, 20.5)
plt.show()