import random
import math

class TestDataGenerator():
    
    # constructor to initialize the number of points and the radius of the circle (optional) as the default radius is 16383 (half of the maximum value of 32767 for x and y coordinates).
    def __init__(self,num_points,radius=16383):
        self.num_points = num_points
        self.radius = radius
        
    # generate random points in 2D space
    def random_points(self):
        rand_points = []
        for _ in range(self.num_points):
            x = random.randint(0,32767)
            y = random.randint(0,32767)
            rand_points.append((x,y))
        return rand_points
    
    # generate polygon points, where the points are vertices of a polygon
    def generate_polygon_points(self, polygon_sides):
        if polygon_sides < 3:
            raise ValueError("A polygon must have at least 3 sides.")
        
        polygon_points = []
        centre_x = 16383
        centre_y = 16383

        for i in range(polygon_sides):
            angle = 2 * math.pi * i / polygon_sides
            x = int(round(centre_x + self.radius * math.cos(angle)))
            y = int(round(centre_y + self.radius * math.sin(angle)))
            polygon_points.append((x, y))

        return polygon_points
    
    # generate circle points, where the points are on the circumference of a circle - works about first 500-600 points due to rounding (that all the numbers had to be integers). If less than 20 points, it resembles a polygon.
    def generate_circle_points(self):
        circle_points = []
        centre_x = 16383
        centre_y = 16383

        for i in range(self.num_points):
            angle = 2 * math.pi * i / self.num_points
            x = int(round(centre_x + self.radius * math.cos(angle)))
            y = int(round(centre_y + self.radius * math.sin(angle)))
            circle_points.append((x, y))

        return circle_points
    
    # generate collinear points
    def generate_collinear_points(self):
        grad = random.randint(-5,10)
        x = 0
        y = 0
        if grad == 0:
            y = 16383
        elif grad > 0:
            y = 2
            x = 2
        else:
            y = 32765
            x = 2    

        rand_points = []
        for _ in range(self.num_points):
            rand_points.append((x,y))
            x += 1
            y += grad  # increment y by the gradient
        return rand_points
   
   
    # check if a point is inside a polygon - used for controlled point generation
    def point_inside_polygon(self, x, y, polygon_points):
        n = len(polygon_points)
        inside = False

        p1x, p1y = polygon_points[0]
        for i in range(n + 1):
            p2x, p2y = polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            x_intersects = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_intersects:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside 
    # controlled point generation for testing (random points inside a polygon)
    def controlled_point_generation(self, polygon_sides):
        if polygon_sides < 3:
            raise ValueError("A polygon must have at least 3 sides.")
        elif self.num_points < polygon_sides:
            raise ValueError("The number of points must be greater than or equal to the number of sides of the polygon.")
        number_of_points_in_shape = self.num_points - polygon_sides
        
        vertices = self.generate_polygon_points(polygon_sides)
        # Compute the centroid of the polygon
        centroid = [sum(x) / len(x) for x in zip(*vertices)]

        points = []
        for _ in range(number_of_points_in_shape):
            # Generate a random angle
            angle = 2 * math.pi * random.random()

            # Generate a random radius
            radius = self.radius * math.sqrt(random.random())

            # Compute the coordinates of the point
            x = centroid[0] + radius * math.cos(angle)
            y = centroid[1] + radius * math.sin(angle)
            
            x = int(round(x))
            y = int(round(y))

            # Check if the point lies inside the polygon
            if self.point_inside_polygon(x, y, vertices):
                points.append((x, y))
        points += vertices
        return points 