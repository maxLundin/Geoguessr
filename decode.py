def decode(scores, geometry, confidence_threshold):
    (num_rows, num_cols) = scores.shape[2:4]
    confidences = []
    rects = []
    baggage = []

    for y in range(0, num_rows):
        scores_data = scores[0, 0, y]
        d_top = geometry[0, 0, y]
        d_right = geometry[0, 1, y]
        d_bottom = geometry[0, 2, y]
        d_left = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(0, num_cols):
            if scores_data[x] < confidence_threshold:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = angles_data[x]
            confidences.append(float(scores_data[x]))

            upper_right = (offsetX + d_right[x], offsetY - d_top[x])
            lower_right = (offsetX + d_right[x], offsetY + d_bottom[x])
            upper_left = (offsetX - d_left[x], offsetY - d_top[x])
            lower_left = (offsetX - d_left[x], offsetY + d_bottom[x])

            rects.append([
                int(upper_left[0]),
                int(upper_left[1]),
                int(lower_right[0] - upper_left[0]),
                int(lower_right[1] - upper_left[1])
            ])

            baggage.append({
                "offset": (offsetX, offsetY),
                "angle": angle,
                "upper_right": upper_right,
                "lower_right": lower_right,
                "upper_left": upper_left,
                "lower_left": lower_left,
                "d_top": d_top[x],
                "d_right": d_right[x],
                "d_bottom": d_bottom[x],
                "d_left": d_left[x]
            })

    return rects, confidences, baggage
