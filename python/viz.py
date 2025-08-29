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
        #return all(cv2.pointPolygonTest(contour, tuple(pt), False) >= 0 for pt in rect)
        return all(cv2.pointPolygonTest(contour, (int(pt[0]), int(pt[1])), False) >= 0 for pt in rect)

    # Function to create a rectangle with additional points along each edge
    def create_rectangle(cX, cY, w, h, x = 5):
        """
        Create a rectangle centered at (cX, cY) with width `w` and height `h`, 
        and add `x` number of points along each edge.
        
        Args:
        - cX, cY: center of the rectangle
        - w, h: width and height of the rectangle
        - x: number of additional points to add along each edge

        Returns:
        - np.array of rectangle vertices including the additional points
        """
        # Calculate corner points
        top_left = [cX - w // 2, cY - h // 2]
        top_right = [cX + w // 2, cY - h // 2]
        bottom_left = [cX - w // 2, cY + h // 2]
        bottom_right = [cX + w // 2, cY + h // 2]
        
        # Generate additional points
        top_edge = [top_left + (i / (x + 1)) * (np.array(top_right) - np.array(top_left)) for i in range(1, x + 1)]
        right_edge = [top_right + (i / (x + 1)) * (np.array(bottom_right) - np.array(top_right)) for i in range(1, x + 1)]
        bottom_edge = [bottom_right + (i / (x + 1)) * (np.array(bottom_left) - np.array(bottom_right)) for i in range(1, x + 1)]
        left_edge = [bottom_left + (i / (x + 1)) * (np.array(top_left) - np.array(bottom_left)) for i in range(1, x + 1)]
        
        # Combine all points into one array
        points = np.array([top_left] + top_edge + [top_right] + right_edge + [bottom_right] + bottom_edge + [bottom_left] + left_edge, dtype=np.int32)
        
        return points

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


def get_random_point_in_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    while True:
        px = random.randint(x, x + w)
        py = random.randint(y, y + h)
        if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
            return px, py
        
def save_room_and_rectangle_image_old(room_id, contour, rectangle2, image_size=(400, 400), background_image=None):
    """
    Save an image of the room contour and the rectangle, optionally drawing over an existing image.
    
    Args:
    - room_id: The ID of the room.
    - contour: The contour of the room as a list of points.
    - rectangle2: A tuple (cX, cY, w, h) representing the rectangle's center and dimensions.
    - image_size: The size of the new image to create if no background_image is provided (width, height).
    - background_image: An optional image to draw over. If provided, it should be a numpy array.
    """
    
    if background_image is not None:
        # Use the provided image as the base
        image = background_image.copy()
    else:
        # Create a blank image
        image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    # Draw the contour of the room
    cv2.drawContours(image, [contour], -1, (0, 0, 0), 2)

    # Draw the rectangle
    cX, cY, w, h = rectangle2
    top_left = (int(cX - w // 2), int(cY - h // 2))
    bottom_right = (int(cX + w // 2), int(cY + h // 2))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 1)

    # draw the center 
    # Center coordinates 
    center_coordinates = (int(cX), int(cY)) 
    
    # Radius of circle 
    radius = 2
    
    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 2 px 
    thickness = 2
    
    # Using cv2.circle() method 
    # Draw a circle with blue line borders of thickness of 2 px 
    cv2.circle(image, center_coordinates, radius, color, thickness) 

    # Save the image
    cv2.imwrite(f"imgs/{room_id}.png", image)
    return image

def save_room_and_rectangle_image(room_id, contour, rectangle2, image_size=(400, 400), background_image=None):
    """
    Save an image of the room contour and the rectangle, drawing the contour only on white pixels.
    
    Args:
    - room_id: The ID of the room.
    - contour: The contour of the room as a list of points.
    - rectangle2: A tuple (cX, cY, w, h) representing the rectangle's center and dimensions.
    - image_size: The size of the new image to create if no background_image is provided (width, height).
    - background_image: An optional image to draw over. If provided, it should be a numpy array.
    
    Returns:
    - image: The resulting image with the contour and rectangle drawn.
    """
    
    if background_image is not None:
        # Use the provided image as the base
        image = background_image.copy()
    else:
        # Create a blank image
        image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    # Create a mask for the contour
    mask = np.zeros_like(image)
    
    # Draw the contour on the mask
    cv2.drawContours(mask, [contour], -1, (1, 1, 1), 2)

    # Iterate over the image and apply the contour where pixels are white
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if np.array_equal(image[y, x], [255, 255, 255]) and not np.array_equal(mask[y, x], [0, 0, 0]):
                image[y, x] = mask[y, x]

    # Draw the rectangle
    cX, cY, w, h = rectangle2
    w = w - 2
    h = h - 2
    top_left = (int(cX - w // 2), int(cY - h // 2))
    bottom_right = (int(cX + w // 2), int(cY + h // 2))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 1)

    # Save the image
    cv2.imwrite(f"imgs/room_{room_id}.png", image)
    return image

def draw_masks(masks, real_nodes, im_size=256, floorplan_id=0):
    #print('masks: ', masks)
    #print('real_nodes: ', real_nodes)
    #exit()
    # process polygons
    polygons = []
    room_centers = []
    contour_areas = []
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
    dwg = svgwrite.Drawing(f'floorplans/floorplan{floorplan_id}.svg', (256, 256))
    rooms = []
    final_img = None
    #rooms_ids_points = []
    for nd, contours in zip(real_nodes, polygons):
        # pick color
        color = ID_COLOR[nd]
        r, g, b = webcolors.hex_to_rgb(color)
        if nd not in [15, 17]:
            new_contours = _fix(contours) 
            new_contours = [c for c in new_contours if cv2.contourArea(c) >= 4] # filter out small contours
            #print('draw room: color = ', color, 'color_id = ', nd, '\n', new_contours) #, '\n', new_contours
            _draw_polygon(dwg, new_contours, color)
            rooms.append(new_contours)

            
            # Calculate center of each room
            for contour in new_contours:
                M = cv2.moments(contour)
                
                best_w, best_h, best_cX, best_cY, best_i = 0, 0, 0, 0, -1

                area = cv2.contourArea(contour) / 100.0  # Scale the area appropriately
                contour_areas.append({"room_id": len(rooms), "color_id": nd, "area": area})


                # initilize with rectangle from room center
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Save the room and rectangle to an image file
                    #rectangle = (best_cX, best_cY, best_w, best_h)
                    #final_img = save_room_and_rectangle_image("id"+str(len(rooms))+"_c"+str(nd)+"_"+str(best_i), contour, rectangle)

                    if nd in [1, 3, 7]:
                        best_w, best_h, best_cX, best_cY = expand_rectangle(contour, cX, cY)

                        # try 10 random centers and expand rectangles from them if there is a better rectangle
                        for i in range(10):
                            cX, cY = get_random_point_in_contour(contour)
                            w, h, cX, cY = expand_rectangle(contour, cX, cY)
                            if w * h > best_w * best_h:
                                best_w, best_h, best_cX, best_cY = w, h, cX, cY
                                best_i = i
                    
                    # Save the room and rectangle to an image file
                    #rectangle = (cX, cY, w, h)
                    #save_room_and_rectangle_image("id"+str(len(rooms))+"_c"+str(nd)+"_"+str(i), contour, rectangle, background_image=final_img)
                
                rectangle = (best_cX, best_cY, best_w, best_h)
                final_img = save_room_and_rectangle_image_old("FP_"+str(floorplan_id)+"_room_id"+str(len(rooms))+"_c"+str(nd)+"_"+str(best_i), contour, rectangle, background_image=final_img)
                
                print("room_id: ", len(rooms), ', color_id = ', nd, ' -------------------> best_i == ', best_i)
                # Calculate minimum bounding rectangle within each room
                rect = cv2.minAreaRect(contour)
                (x, y), (rect_w, rect_h), angle = rect


                # Calculate bounding rectangle to determine max width and height
                x, y, w, h = cv2.boundingRect(contour)


                if M["m00"] != 0 and w > 0 and h > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    rect_to_total_area = round((best_w*best_h) / (w*h), 2)
                    room_centers.append({"center": (cX/10, cY/10), "color_id": nd, "room_id": len(rooms), "max_w": w/10, "max_h": h/10, "best_w":best_w/10, "best_h":best_h/10, "best_cX":best_cX/10, "best_cY" : best_cY/10, "best_ratio": rect_to_total_area})
                else:
                    room_centers.append({"center": None, "color_id": nd, "room_id": len(rooms), "max_w": w/10, "max_h": h/10, "max_rect_width": 0, "max_rect_height": 0})
            
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
            #print('draw door: color = ', color, 'color_id = ', nd, '\n', contours)
            _draw_polygon(dwg, [contour], color, with_stroke=False)

            # Convert each point in the contour to a tuple (x, y)
            points = [(point[0][0]/10, point[0][1]/10) for point in contour]
            doors_points.append(points)


    # Save contours to JSON
    #json_filename = "contour_points.json"
    #with open(json_filename, 'w') as f:
    #    json.dump(contour_points, f, cls=CustomEncoder)

    # Save contours to JSON
    json_filename = "floorplans/rooms_points_"+str(floorplan_id)+".json"
    with open(json_filename, 'w') as f:
        json.dump([rooms_points, doors_points, room_centers, contour_areas], f, cls=CustomEncoder)

    dwg.save()
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