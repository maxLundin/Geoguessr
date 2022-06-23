import math


def rects2polys(rects, thetas, origins):
    polygons = []
    for i, box in enumerate(rects):
        upper_left_x = box[0]
        upper_left_y = box[1]
        lower_right_x = box[0] + box[2]
        lower_right_y = box[1] + box[3]

        points = [
            (upper_left_x, upper_left_y),
            (lower_right_x, upper_left_y),
            (lower_right_x, lower_right_y),
            (upper_left_x, lower_right_y)
        ]

        # the offset is the point at which the rectangle is rotated
        rotation_point = (int(origins[i][0]), int(origins[i][1]))

        polygons.append(rotate_points(points, thetas[i], rotation_point))

    return polygons


def rotate_points(points, theta, origin):
    rotated = []
    for xy in points:
        rotated.append(rotate_around_point(xy, theta, origin))

    return rotated


def rotate_around_point(xy, radians, origin=(0, 0)):
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy