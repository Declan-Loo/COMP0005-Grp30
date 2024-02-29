import random
import math
import timeit
import matplotlib.pyplot as plt

def orientation(p, q, r):
    # Determine the orientation of the triplet (p, q, r).
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    if val > 0:
        return 1
    else:
        return 2  # Clockwise or counterclockwise


def jarvismarch(inputSet):
    '''
    Returns the list of points that lie on the convex hull (jarvis march algorithm)
            Parameters:
                    inputSet (list): a list of 2D points

            Returns:
                    outputSet (list): a list of 2D points
    '''
    # Determine the orientation of the triplet (p, q, r).

    # Check if there are at least 3 points
    if len(inputSet) < 3:
        return inputSet
    # Find the leftmost point
    leftmost = min(inputSet, key=lambda point: (point[0], point[1]))
    outputSet = []
    p = leftmost

    while True:
        outputSet.append(p)
        if inputSet[0] != p:
            q = inputSet[0]
        else:
            q = inputSet[1]
        for r in inputSet:
            if orientation(p, q, r) == 2:  # If r is more counterclockwise than current q
                q = r
        p = q  # q is the most counterclockwise with respect to p
        if p == leftmost:  # Closed the loop
            break

    return outputSet


def grahamscan(points):
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


def chen(points):
    def divideAndConquer(points, m):
        # Split points into subsets.
        subsets = [points[i:i + m] for i in range(0, len(points), m)]
        return [grahamscan(subset) for subset in subsets]

    # m represents the number of points in each subset for the divide-and-conquer approach.
    # The value is based on the logarithm of the total number of input points to balance efficiency.
    # We use max(3, ...) to ensure m is at least 3 since a convex hull requires at least three points to form a valid shape.
    m = max(3, int(math.ceil(math.log(len(points), 2))))
    hulls = divideAndConquer(points, m)
    # Flatten the list of hulls into a single list of points
    mergedHulls = []
    for hull in hulls:  # For each sublist (hull) in the list of lists (hulls)
        for point in hull:  # For each item (point) in the sublist (hull)
            mergedHulls.append(point)  # Add the item (point) to the new list (mergedHulls)
    outputSet = jarvismarch(mergedHulls)
    return outputSet


import random
import math

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
        points = data_generator.random_points()
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
            jarvis_time += time_algorithm(jarvismarch, points)
            graham_time += time_algorithm(grahamscan, points)
            chan_time += time_algorithm(chen, points)
        jarvis_trials.append(jarvis_time/n)
        graham_trials.append(graham_time/n)
        chan_trials.append(chan_time/n)
        
    return jarvis_trials, graham_trials, chan_trials
        
    
point_range = [i for i in range(100, 10000, 1000)]
jarvis_trials, graham_trials, chan_trials = trials(point_range, 50)

plt.plot(point_range, jarvis_trials, label = "Jarvis March", color = 'red')
plt.plot(point_range, graham_trials, label = "Graham Scan", color = 'blue')
plt.plot(point_range, chan_trials, label = "Chan's Algorithm" , color = 'green')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(20.5, 20.5)
plt.show()