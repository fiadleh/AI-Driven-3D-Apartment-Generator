import argparse
import os
import numpy as np
import math
import sys
import random
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
# import matplotlib.pyplot as plt
# import networkx as nx
import glob
import cv2
import webcolors
import time
import svgwrite
import json

ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7, "study room": 8,
              "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}

CLASS_ROM = {}
for x, y in ROOM_CLASS.items():
    CLASS_ROM[y] = x
ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8', 6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 15: '#727171', 16: '#785A67', 17: '#D3A2C7'}

def pad_im(cr_im, final_size=256, bkg_color='white'):    
    new_size = int(np.max([np.max(list(cr_im.size)), final_size]))
    padded_im = Image.new('RGB', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])//2, (new_size-cr_im.size[1])//2))
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im

def draw_graph(g_true):
    # build true graph 
    G_true = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    
    # add nodes
    for k, label in enumerate(g_true[0]):
        _type = label
        if _type >= 0 and _type not in [15, 17]:
            G_true.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            edgecolors.append('blue')
            linewidths.append(0.0)
            
    # add outside node
    G_true.add_nodes_from([(-1, {'label':-1})])
    colors_H.append("white")
    node_size.append(750)
    edgecolors.append('black')
    linewidths.append(3.0)
    
    # add edges
    for k, m, l in g_true[1]:
        _type_k = g_true[0][k]
        _type_l = g_true[0][l]
        if m > 0 and (_type_k not in [15, 17] and _type_l not in [15, 17]):
            G_true.add_edges_from([(k, l)])
            edge_color.append('#D3A2C7')
        elif m > 0 and (_type_k==15 or _type_l==15) and (_type_l!=17 and _type_k!=17):
            if _type_k==15:
                G_true.add_edges_from([(l, -1)])   
            elif _type_l==15:
                G_true.add_edges_from([(k, -1)])
            edge_color.append('#727171')
    plt.figure()
    pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
    nx.draw(G_true, pos, node_size=node_size, linewidths=linewidths, node_color=colors_H, font_size=14, font_color='white',\
            font_weight='bold', edgecolors=edgecolors, edge_color=edge_color, width=4.0, with_labels=False)
    plt.tight_layout()
    plt.savefig('./dump/_true_graph.jpg', format="jpg")
    plt.close('all')
    rgb_im = Image.open('./dump/_true_graph.jpg')
    rgb_arr = pad_im(rgb_im).convert('RGBA')
    return G_true, rgb_im

def _snap(polygons, ths=[2, 4]):
    polygons = list(polygons)
    cs = np.concatenate([np.stack(p) for ps in polygons for p in ps], 0).reshape(-1, 2)
    new_cs = np.array(cs)
    for th in ths:
        for i in range(len(new_cs)):
            x0, y0 = new_cs[i]
            x0_avg, y0_avg = [], []
            tracker = []
            for j in range(len(new_cs)):
                x1, y1 = new_cs[j]

                # horizontals
                if abs(x1-x0) <= th:
                    x0_avg.append(x1)
                    tracker.append((j, 0))
                # verticals
                if abs(y1-y0) <= th:
                    y0_avg.append(y1)
                    tracker.append((j, 1))
            avg_vec = [np.mean(x0_avg), np.mean(y0_avg)]

            # set others
            for k, m in tracker:
                new_cs[k, m] = avg_vec[m]

    # create map to new corners
    c_map = {}
    for c, new_c in zip(cs, new_cs):
        c_map[tuple(c)] = tuple(new_c)

    # update polygons
    for i in range(len(polygons)):
        for j in range(len(polygons[i])):
            for k in range(len(polygons[i][j])):
                xy = polygons[i][j][k][0]
                polygons[i][j][k][0] = np.array(c_map[tuple(xy)])
    return polygons

def _fix(contours):

    # fill old contours
    m_cv = np.full((256, 256, 1), 0).astype('uint8')
    cv2.fillPoly(m_cv, pts=contours, color=(255, 255, 255))

    # erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m_cv = cv2.erode(m_cv, kernel)
    
    # dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m_cv = cv2.dilate(m_cv, kernel)

    # get new contours
    ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
    new_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return new_contours

def _assign_door(contours, rooms):

    # define doors
    horizontal_door = np.array([[[-8, -2]], [[8, -2]], [[8, 2]], [[-8, 2]]]) # (width, height)
    vertical_door = np.array([[[-2, -8]], [[2, -8]], [[2, 8]], [[-2, 8]]]) # (width, height)
    unit = np.array([[[-1, -1]], [[1, -1]], [[1, 1]], [[-1, 1]]]) # debug

    # assign door to room
    door = np.concatenate(contours, 0)[:, 0, :]
    door_mean = np.mean(door, 0)
    rooms = np.concatenate([r for rs in rooms for r in rs], 0)[:, 0, :]
    dist = np.sum((rooms - door_mean)**2, axis=1)
    idx = np.argmin(dist)
    pt = rooms[idx]

    # determine vertical/horizontal
    wh = np.max(door, 0)-np.min(door, 0)
    if wh[0] > wh[1]: # horizontal
        new_door = horizontal_door + np.array([door_mean[0], pt[1]]).astype('int')
    else: # vertical
        new_door = vertical_door + np.array([pt[0], door_mean[1]]).astype('int')
    return new_door

def _draw_polygon(dwg, contours, color, with_stroke=True):
    for c in contours:
        pts = [(float(c[0]), float(c[1])) for c in c[:, 0, :]]
        if with_stroke:
            dwg.add(dwg.polygon(pts, stroke='black', stroke_width=4, fill=color))
        else:
            dwg.add(dwg.polygon(pts, stroke='none', fill=color))
    return


# Using json.JSONEncoder for customization
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super().default(obj)

def expand_rectangle(contour, cX, cY):
    w, h = 1, 1  # Start with a rectangle of 1x1

    def fits_inside(rect):
        return all(cv2.pointPolygonTest(contour, tuple(pt), False) >= 0 for pt in rect)

    # Function to create a rectangle based on center and dimensions
    def create_rectangle(cX, cY, w, h):
        return np.array([
            [cX - w // 2, cY - h // 2],
            [cX + w // 2, cY - h // 2],
            [cX - w // 2, cY + h // 2],
            [cX + w // 2, cY + h // 2]
        ], dtype=np.int32)

    directions = ['right', 'left', 'down', 'up']
    expanding = True
    while expanding:
        expanding = False

        # Try expanding in each direction
        for direction in directions:
            if direction == 'right':
                temp_w = w + 1
                temp_rect = create_rectangle(cX + 0.5, cY, temp_w, h)
                if fits_inside(temp_rect):
                    cX += 0.5  # Update center
                    w = temp_w
                    expanding = True

            elif direction == 'left':
                temp_w = w + 1
                temp_rect = create_rectangle(cX - 0.5, cY, temp_w, h)
                if fits_inside(temp_rect):
                    cX -= 0.5  # Update center
                    w = temp_w
                    expanding = True

            elif direction == 'down':
                temp_h = h + 1
                temp_rect = create_rectangle(cX, cY + 0.5, w, temp_h)
                if fits_inside(temp_rect):
                    cY += 0.5  # Update center
                    h = temp_h
                    expanding = True

            elif direction == 'up':
                temp_h = h + 1
                temp_rect = create_rectangle(cX, cY - 0.5, w, temp_h)
                if fits_inside(temp_rect):
                    cY -= 0.5  # Update center
                    h = temp_h
                    expanding = True

    # Final rectangle dimensions
    return w, h, cX, cY

def expand_rectangle22222222(contour, cX, cY):
    max_w, max_h = 1, 1

    # Function to check if rectangle fits inside the contour
    def fits_inside(rect):
        return all(cv2.pointPolygonTest(contour, tuple(pt), False) >= 0 for pt in rect)

    # Expand width to the right
    while True:
        right_rect = np.array([
            [cX, cY - max_h // 2],
            [cX + max_w, cY - max_h // 2],
            [cX, cY + max_h // 2],
            [cX + max_w, cY + max_h // 2]
        ], dtype=np.int32)
        if fits_inside(right_rect):
            max_w += 1
        else:
            break

    # Expand width to the left
    max_w -= 1
    while True:
        left_rect = np.array([
            [cX - max_w, cY - max_h // 2],
            [cX, cY - max_h // 2],
            [cX - max_w, cY + max_h // 2],
            [cX, cY + max_h // 2]
        ], dtype=np.int32)
        if fits_inside(left_rect):
            max_w += 1
        else:
            break

    # Final maximum width
    max_w -= 1
    final_w = 2 * max_w

    # Expand height downward
    while True:
        down_rect = np.array([
            [cX - max_w // 2, cY],
            [cX + max_w // 2, cY],
            [cX - max_w // 2, cY + max_h],
            [cX + max_w // 2, cY + max_h]
        ], dtype=np.int32)
        if fits_inside(down_rect):
            max_h += 1
        else:
            break

    # Expand height upward
    max_h -= 1
    while True:
        up_rect = np.array([
            [cX - max_w // 2, cY - max_h],
            [cX + max_w // 2, cY - max_h],
            [cX - max_w // 2, cY],
            [cX + max_w // 2, cY]
        ], dtype=np.int32)
        if fits_inside(up_rect):
            max_h += 1
        else:
            break

    # Final maximum height
    max_h -= 1
    final_h = 2 * max_h

    return final_w, final_h, cX, cY

def get_random_point_in_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    while True:
        px = random.randint(x, x + w)
        py = random.randint(y, y + h)
        if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
            return px, py
        
def save_room_and_rectangle_image(room_id, contour, rectangle2, image_size=(800, 800)):
    # Create a blank image
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255

    # Draw the contour of the room
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # Draw the rectangle
    cX, cY, w, h = rectangle2
    top_left = (int(cX - w // 2), int(cY - h // 2))
    bottom_right = (int(cX + w // 2), int(cY + h // 2))
    #print("---------------- ", (type(image), type(top_left[0]), type(bottom_right), type((255, 0, 0)), type(2)))
    cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

    # Save the image
    cv2.imwrite(f"imgs/room_{room_id}.png", image)

def draw_masks(masks, real_nodes, im_size=256):
    #print('masks: ', masks)
    #print('real_nodes: ', real_nodes)
    #exit()
    # process polygons
    polygons = []
    room_centers = []
    for m, nd in zip(masks, real_nodes):
        # resize map
        m[m>0] = 255
        m[m<0] = 0
        m_lg = cv2.resize(m, (im_size, im_size), interpolation = cv2.INTER_NEAREST) 

        # extract contour
        m_cv = m_lg[:, :, np.newaxis].astype('uint8')
        ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if len(c) > 0]
        polygons.append(contours)

        

    polygons = _snap(polygons)
    #print("polygons: ", polygons)

    

    # draw rooms polygons
    dwg = svgwrite.Drawing('./floorplan.svg', (256, 256))
    rooms = []
    #rooms_ids_points = []
    for nd, contours in zip(real_nodes, polygons):
        # pick color
        color = ID_COLOR[nd]
        r, g, b = webcolors.hex_to_rgb(color)
        if nd not in [15, 17]:
            new_contours = _fix(contours) 
            new_contours = [c for c in new_contours if cv2.contourArea(c) >= 4] # filter out small contours
            print('draw room: color = ', color, 'color_id = ', nd, '\n', new_contours)
            _draw_polygon(dwg, new_contours, color)
            rooms.append(new_contours)

            # Calculate center of each room
            for contour in new_contours:
                M = cv2.moments(contour)

                best_w, best_h, best_cX, best_cY = 0, 0, 0, 0

                for i in range(10):
                    cX, cY = get_random_point_in_contour(contour)
                    w, h, cX, cY = expand_rectangle(contour, cX, cY)
                    if w * h > best_w * best_h:
                        best_w, best_h, best_cX, best_cY = w, h, cX, cY
                    
                    # Save the room and rectangle to an image file
                    rectangle = (cX, cY, w, h)
                    #print("ffffffffffffffffffffffffffff ",(cX, cY, w, h), rectangle)
                    #exit()
                    save_room_and_rectangle_image(str(nd)+"_"+str(i), contour, rectangle)

                # Calculate minimum bounding rectangle within each room
                rect = cv2.minAreaRect(contour)
                (x, y), (rect_w, rect_h), angle = rect
                max_rect_width, max_rect_height = max(rect_w, rect_h), min(rect_w, rect_h)


                # Calculate bounding rectangle to determine max width and height
                x, y, w, h = cv2.boundingRect(contour)
                width_greater_than_height = w > h

                max_width, max_height = 0, 0


                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Initialize the maximum width and height
                    max_w, max_h = 1, 1

                    # Expand width to the right
                    while True:
                        right_rect = np.array([
                            [cX, cY - max_h // 2],
                            [cX + max_w, cY - max_h // 2],
                            [cX, cY + max_h // 2],
                            [cX + max_w, cY + max_h // 2]
                        ], dtype=np.int32)

                        if all(cv2.pointPolygonTest(contour, tuple(pt), False) >= 0 for pt in right_rect):
                            max_w += 1
                        else:
                            break

                    # Expand width to the left
                    max_w -= 1  # reset to last valid width
                    while True:
                        left_rect = np.array([
                            [cX - max_w, cY - max_h // 2],
                            [cX, cY - max_h // 2],
                            [cX - max_w, cY + max_h // 2],
                            [cX, cY + max_h // 2]
                        ], dtype=np.int32)

                        if all(cv2.pointPolygonTest(contour, tuple(pt), False) >= 0 for pt in left_rect):
                            max_w += 1
                        else:
                            break

                    # Final maximum width is twice the radius (left and right)
                    max_w -= 1  # reset to last valid width
                    final_w = 2 * max_w

                    # Expand height downward
                    while True:
                        down_rect = np.array([
                            [cX - max_w // 2, cY],
                            [cX + max_w // 2, cY],
                            [cX - max_w // 2, cY + max_h],
                            [cX + max_w // 2, cY + max_h]
                        ], dtype=np.int32)

                        if all(cv2.pointPolygonTest(contour, tuple(pt), False) >= 0 for pt in down_rect):
                            max_h += 1
                        else:
                            break

                    # Expand height upward
                    max_h -= 1  # reset to last valid height
                    while True:
                        up_rect = np.array([
                            [cX - max_w // 2, cY - max_h],
                            [cX + max_w // 2, cY - max_h],
                            [cX - max_w // 2, cY],
                            [cX + max_w // 2, cY]
                        ], dtype=np.int32)

                        if all(cv2.pointPolygonTest(contour, tuple(pt), False) >= 0 for pt in up_rect):
                            max_h += 1
                        else:
                            break

                    # Final maximum height is twice the radius (up and down)
                    max_h -= 1  # reset to last valid height
                    final_h = 2 * max_h
                    max_w = final_w
                    max_h = final_h

                    room_centers.append({"center": (cX/10, cY/10), "color_id": nd, "room_id": len(rooms), "max_w": w, "max_h": h, "max_rect_width": max_w, "max_rect_height": max_h, "best_w":best_w, "best_h":best_h, "best_cX":best_cX/10, "best_cY" : best_cY/10})
                else:
                    room_centers.append({"center": None, "color_id": nd, "room_id": len(rooms), "max_w": w, "max_h": h, "max_rect_width": 0, "max_rect_height": 0})
            #rooms_ids_points.append({"id": int(nd), "contours": list(new_contours)})
    #print("rooms: ", rooms)
    #y = json.dumps(rooms_ids_points, cls=CustomEncoder)

    #for i in range(rooms):
    #with open('./result.json', 'w') as fp:
    #    json.dump(y, fp)

    
    

    # Prepare contour points for JSON
    contour_points = []
    for contours in polygons:
        for contour in contours:
            # Convert each point in the contour to a tuple (x, y)
            points = [(point[0][0]/10, point[0][1]/10) for point in contour]
            contour_points.append(points)

    rooms_points = []
    for contours in rooms:
        for contour in contours:
            # Convert each point in the contour to a tuple (x, y)
            points = [(point[0][0]/10, point[0][1]/10) for point in contour]
            rooms_points.append(points)

    

    doors_points = []

    # draw doors
    for nd, contours in zip(real_nodes, polygons):
        # pick color
        color = ID_COLOR[nd]
        if nd in [15, 17] and len(contours) > 0:
            contour = _assign_door([contours[0]], rooms)
            #print('draw door: color = ', color, 'color_id = ', nd, '\n', contour)
            _draw_polygon(dwg, [contour], color, with_stroke=False)

            # Convert each point in the contour to a tuple (x, y)
            points = [(point[0][0]/10, point[0][1]/10) for point in contour]
            doors_points.append(points)


    # Save contours to JSON
    #json_filename = "contour_points.json"
    #with open(json_filename, 'w') as f:
    #    json.dump(contour_points, f, cls=CustomEncoder)

    # Save contours to JSON
    json_filename = "rooms_points.json"
    with open(json_filename, 'w') as f:
        json.dump([rooms_points, doors_points, room_centers], f, cls=CustomEncoder)

    return dwg.tostring()

## OLD CODE -- BACKUP
# def draw_masks(masks, real_nodes, im_size=256):
    
#     # process polygons
#     polygons = []
#     for m, nd in zip(masks, real_nodes):
#         # resize map
#         m[m>0] = 255
#         m[m<0] = 0
#         m_lg = cv2.resize(m, (im_size, im_size), interpolation = cv2.INTER_NEAREST) 

#         # extract contour
#         m_cv = m_lg[:, :, np.newaxis].astype('uint8')
#         ret, thresh = cv2.threshold(m_cv, 127, 255, 0)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         contours = [c for c in contours if len(c) > 0]
#         polygons.append(contours)
#     polygons = _snap(polygons)

#     # draw rooms polygons
#     dwg = svgwrite.Drawing('./floorplan.svg', (256, 256))
#     bg_img = np.full((256, 256, 3), 255).astype('uint8')
#     rooms = []
#     for nd, contours in zip(real_nodes, polygons):
#         # pick color
#         color = ID_COLOR[nd]
#         r, g, b = webcolors.hex_to_rgb(color)
#         if nd not in [15, 17]:
#             new_contours = _fix(contours) 
#             new_contours = [c for c in new_contours if cv2.contourArea(c) >= 4] # filter out small contours
#             cv2.fillPoly(bg_img, pts=new_contours, color=(r, g, b, 255))
#             cv2.drawContours(bg_img, new_contours, -1, (0, 0, 0, 255), 2)
#             _draw_polygon(dwg, new_contours, color)
#             rooms.append(new_contours)

#     # draw doors
#     for nd, contours in zip(real_nodes, polygons):
#         if nd in [15, 17] and len(contours) > 0:
#             # cv2.fillPoly(bg_img, pts=[contours[0]], color=(0, g, 0, 255))
#             contour = _assign_door([contours[0]], rooms)
#             cv2.fillPoly(bg_img, pts=[contour], color=(r, g, b, 255))
#             _draw_polygon(dwg, [contour], color, with_stroke=False)
#     bg_img = Image.fromarray(bg_img)
#     return dwg.tostring()