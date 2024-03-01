import random
import matplotlib.pyplot as plt
import timeit
import math

# find the orientation of three 2D points - to see if the rotation from points is counterclockwise or clockwise based on the middle point (p2)
def orientation(p1 : tuple[int,int],p2 : tuple[int,int] ,p3: tuple[int,int]) -> int:
        # find the difference of the gradients between p3 and p2, and p2 and p1, to check the orientation
        grad_diff = ((p3[1]-p2[1])*(p2[0]-p1[0])) - ((p2[1]-p1[1])*(p3[0]-p2[0]))
        
        # check if the orientation is counterclockwise (so return 1) - as we prefer the orientation to be counterclockwise in the Jarvis March algorithm.
        if grad_diff > 0:
                return 1
        # check if the orientation is clockwise (so return -1)
        elif grad_diff < 0:
                return -1
        # otherwise the orientation will be collinear (so return 0) - so there is no slope but a straight line
        else:
                return 0
def dist(p1 : tuple[int,int], p2: tuple[int,int]) -> float:
        return math.sqrt((p2[1]-p1[1])**2 +(p2[0]-p1[0])**2)

# tutorial: https://www.youtube.com/watch?v=nBvCZi34F_o
def jarvismarch(inputSet : list[tuple[int,int]]) -> list[tuple[int,int]]:
    '''
    Returns the list of points that lie on the convex hull (jarvis march algorithm)
            Parameters:
                    inputSet (list): a list of 2D points

            Returns:
                    outputSet (list): a list of 2D points
    '''
    # find the left-most point
    if len(inputSet) < 3:
        return inputSet

    leftmostPoint = min(inputSet)
    outputSet = []
    while True:
        outputSet.append(leftmostPoint)
        next_point = inputSet[0]
        for point in inputSet:
                rotation = orientation(leftmostPoint, next_point, point)
                # check if the rotation is counterclockwise, where the first point is the next point in the hull, otherwise if rotation is collinear, then check the distance between the leftmost point and the next point, and if the distance is greater than the distance between the leftmost point and the current point, then the next point is the current point.
                if next_point == leftmostPoint or rotation == 1 or (rotation == 0 and dist(leftmostPoint, point) > dist(leftmostPoint,next_point)):
                        next_point = point
        
        # now check for the next point after finding the point that is on the convex hull                
        leftmostPoint = next_point
        # check if the algorithm returns back to the starting point and then end the loop, returning the coordinates of the hull.
        if leftmostPoint == outputSet[0]:
                # convex hull is complete
                break  

    #ADD YOUR CODE HERE

    return outputSet

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
    # positive crossproduct means right turn at p1 (clockwise), negative value means left turn at p1 (counterclockwise), and 0 means collinear
    return (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])

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
    inputSet = sorted(inputSet, key= lambda x: (getPolarAngle(p0, x), getDistance(p0, x)))

    # initialize a stack, outputSet, to store the vertices of the convex hull
    outputSet = [p0, inputSet[1], inputSet[2]]

    for i in range(3, len(inputSet)):
        # while the angle formed by points next-to-top(s), top(s) and p_i makes a nonleft turn, pop from stack
        while len(outputSet) > 1 and crossProduct(outputSet[-2], outputSet[-1], inputSet[i]) <= 0:
            outputSet.pop()
        outputSet.append(inputSet[i])
    return outputSet


def getGroupPoints(inputSet,m):
        ans = []
        n = len(inputSet)
        for i in range(0, n, m):
             # Create a group of 'm' points
            group = inputSet[i:i+m]
            # Apply Graham's scan to the group and extend the answer list
            ans.append(grahamscan(group))
        print(ans)
        return ans
def binarySearch(inputSet,leftPoint,nextPoint):
    Points = []
            
    for subset in inputSet:
        left = 0
        right = len(subset)-1
        
        if len(subset) <= 2:
            return subset[0] if len(subset) == 1 else max(subset, key=lambda p: calAngle(leftPoint, nextPoint, p))
        count = 0
        while left <=right:
            mid = (left + right) // 2
            mid_point = subset[mid]
            print(left,right)
            
            if mid > 0 and mid < len(subset) - 1:
                angle = calAngle(leftPoint, nextPoint, mid_point)
                angle_prev = calAngle(leftPoint, nextPoint, subset[mid - 1])
                angle_next = calAngle(leftPoint, nextPoint, subset[mid + 1])

                if angle > angle_prev and angle > angle_next:
                    Points.append(mid_point)
                    break
                elif angle < angle_next:
                    left = mid + 1
                else:
                    right = mid - 1
            if right == left:
                count += 1
            if count > 0:
                break
                
                
    print(Points,"bin")
    return Points
        
def calAngle(leftPoint, nextPoint, point):
    u = (nextPoint[0] - leftPoint[0], nextPoint[1] - leftPoint[1])
    v = (point[0] - nextPoint[0], point[1] - nextPoint[1])

    length_u = math.sqrt(u[0]**2 + u[1]**2)
    length_v = math.sqrt(v[0]**2 + v[1]**2)

    if length_u == 0 or length_v == 0:
        return 0  

    dot_product = u[0] * v[0] + u[1] * v[1]
    cos_theta = dot_product / (length_u * length_v)
    cos_theta = max(min(cos_theta, 1), -1)
    theta = math.acos(cos_theta)
    angle_in_degrees = math.degrees(theta)

    return angle_in_degrees
    

def nextPoints(leftPoint,nextPoint,Points):
    largestAngle = calAngle(leftPoint,nextPoint,Points[0])
    Point = Points[0]
    for i in Points:
        nextAngle = calAngle(leftPoint,nextPoint,i)
        if largestAngle < nextAngle:
            largestAngle = nextAngle
            Point = i
    print(Point)
    return Point

def chens(inputSet):
    Points = []
    m = 3
    nextPoint = max(inputSet,key=lambda x:x[0])
    leftPoint =(0,nextPoint[1])
    Points.append(nextPoint)
    while m < len(inputSet):
        divided_Points = getGroupPoints(inputSet,m)
        newPoints = []
        while True:
            vertex = binarySearch(divided_Points,leftPoint,nextPoint)
            if not vertex:
                break
            a = nextPoints(leftPoint,nextPoint,vertex)
            if a == Points[0]:
                break
            if a not in Points:
                newPoints.append(a)
                
            leftPoint = nextPoint
            nextPoint = a
        print(newPoints)
        Points.extend(newPoints)
        if Points[-1] == Points[0]:
            break
        m*=2
    return Points


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
            jarvis_time += time_algorithm(jarvismarch, points)
            graham_time += time_algorithm(grahamscan, points)
            chan_time += time_algorithm(chens, points)
        jarvis_trials.append(jarvis_time/n)
        graham_trials.append(graham_time/n)
        chan_trials.append(chan_time/n)
        
    return jarvis_trials, graham_trials, chan_trials
        
    
point_range = [i for i in range(10,10000,100)]
jarvis_trials, graham_trials, chan_times = trials(point_range, 20)

plt.plot(point_range, jarvis_trials, label = "Jarvis", color = 'red')
plt.plot(point_range, graham_trials, label = "Graham", color = 'blue')
plt.plot(point_range, chan_trials, label = "Chan's Algorithm" , color = 'green')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(20.5, 20.5)
plt.show()