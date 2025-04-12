import random
import time
import threading
import pygame
import sys

# Default values of signal timers
defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150
defaultYellow = 5

signals = []
noOfSignals = 4
currentGreen = 0  # Indicates which signal is green currently
nextGreen = (currentGreen + 1) % noOfSignals  # Indicates which signal will turn green next
currentYellow = 0  # Indicates whether yellow signal is on or off

speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5}  # average speeds of vehicles

# Coordinates of vehicles' start
x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
y = {'right': [348, 370, 398], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

vehicles = {'right': {0: [], 1: [], 2: [], 'crossed': 0}, 'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0}, 'up': {0: [], 1: [], 2: [], 'crossed': 0}}
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]
vehicleCountCoods = [(480, 210), (880, 210), (880, 550), (480, 550)]
vehicleCountTexts = ["0", "0", "0", "0"]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

# Gap between vehicles
stoppingGap = 25  # stopping gap
movingGap = 25  # moving gap

initial_red_time = 5

# set allowed vehicle types here
allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}
allowedVehicleTypesList = []
vehiclesTurned = {'right': {1: [], 2: []}, 'down': {1: [], 2: []}, 'left': {1: [], 2: []}, 'up': {1: [], 2: []}}
vehiclesNotTurned = {'right': {1: [], 2: []}, 'down': {1: [], 2: []}, 'left': {1: [], 2: []}, 'up': {1: [], 2: []}}
rotationAngle = 3
mid = {'right': {'x': 705, 'y': 445}, 'down': {'x': 695, 'y': 450}, 'left': {'x': 695, 'y': 425},
       'up': {'x': 695, 'y': 400}}
# set random or default green signal time here
randomGreenSignalTimer = True
# set random green signal time range here
randomGreenSignalTimerRange = [10, 20]

pygame.init()
simulation = pygame.sprite.Group()


def highlight_density_area(vehicles, screen, threshold_distance_meters=5):
    directions = ['right', 'down', 'left', 'up']
    # Stopping coordinates
    defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

    # Initialize dictionaries to hold the bounding box coordinates for each direction
    bounding_boxes = {direction: {'min_x': float('inf'), 'max_x': float('-inf'),
                                  'min_y': float('inf'), 'max_y': float('-inf')}
                      for direction in directions}

    # Iterate through all vehicles in the group
    for vehicle in vehicles:
        # Get vehicle's current direction and position
        direction = vehicle.direction  # Assuming vehicle object has a `direction` attribute
        vehicle_rect = vehicle.currentImage.get_rect(topleft=(vehicle.x, vehicle.y))

        # Update bounding box coordinates based on direction and stopping point
        if direction == 'right':
            # Ensure the bounding box stays visible until the vehicle fully crosses the stop line
            if vehicle_rect.left <= defaultStop['right']:
                bounding_boxes['right']['min_x'] = min(bounding_boxes['right']['min_x'], vehicle_rect.left)
                bounding_boxes['right']['max_x'] = max(bounding_boxes['right']['max_x'], vehicle_rect.right)
                bounding_boxes['right']['min_y'] = min(bounding_boxes['right']['min_y'], vehicle_rect.top)
                bounding_boxes['right']['max_y'] = max(bounding_boxes['right']['max_y'], vehicle_rect.bottom)

        elif direction == 'left':
            if vehicle_rect.right >= defaultStop['left']:
                bounding_boxes['left']['min_x'] = min(bounding_boxes['left']['min_x'], vehicle_rect.left)
                bounding_boxes['left']['max_x'] = max(bounding_boxes['left']['max_x'], vehicle_rect.right)
                bounding_boxes['left']['min_y'] = min(bounding_boxes['left']['min_y'], vehicle_rect.top)
                bounding_boxes['left']['max_y'] = max(bounding_boxes['left']['max_y'], vehicle_rect.bottom)

        elif direction == 'down':
            if vehicle_rect.top <= defaultStop['down']:
                bounding_boxes['down']['min_x'] = min(bounding_boxes['down']['min_x'], vehicle_rect.left)
                bounding_boxes['down']['max_x'] = max(bounding_boxes['down']['max_x'], vehicle_rect.right)
                bounding_boxes['down']['min_y'] = min(bounding_boxes['down']['min_y'], vehicle_rect.top)
                bounding_boxes['down']['max_y'] = max(bounding_boxes['down']['max_y'], vehicle_rect.bottom)

        elif direction == 'up':
            if vehicle_rect.bottom >= defaultStop['up']:
                bounding_boxes['up']['min_x'] = min(bounding_boxes['up']['min_x'], vehicle_rect.left)
                bounding_boxes['up']['max_x'] = max(bounding_boxes['up']['max_x'], vehicle_rect.right)
                bounding_boxes['up']['min_y'] = min(bounding_boxes['up']['min_y'], vehicle_rect.top)
                bounding_boxes['up']['max_y'] = max(bounding_boxes['up']['max_y'], vehicle_rect.bottom)

    # Draw the bounding boxes based on distance threshold
    for direction in directions:
        bbox = bounding_boxes[direction]
        if bbox['min_x'] < bbox['max_x'] and bbox['min_y'] < bbox['max_y']:
            # Calculate the width and height in pixels
            width_pixels = bbox['max_x'] - bbox['min_x']
            height_pixels = bbox['max_y'] - bbox['min_y']

            # Convert width and height to meters using the provided formula
            width_meters = width_pixels * 0.03199692
            height_meters = height_pixels * 0.03199692

            # Draw the bounding box only if it meets the distance threshold in meters
            if width_meters > threshold_distance_meters or height_meters > threshold_distance_meters:
                pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(bbox['min_x'], bbox['min_y'],
                                                                  width_pixels, height_pixels), 3)


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        self.crossedIndex = 0

        path = "D:\\python project\\car project\\Adaptive-Traffic-Signal-Timer-main\\Adaptive-Traffic-Signal-Timer-main\\Code\\YOLO\\darkflow\\images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.image = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

        # Load the image
        self.originalImage = pygame.image.load(path)
        self.image = pygame.image.load(path)

        if (len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0):
            if (direction == 'right'):
                self.stop = vehicles[direction][lane][self.index - 1].stop
                - vehicles[direction][lane][self.index - 1].image.get_rect().width
                - stoppingGap
            elif (direction == 'left'):
                self.stop = vehicles[direction][lane][self.index - 1].stop
                + vehicles[direction][lane][self.index - 1].image.get_rect().width
                + stoppingGap
            elif (direction == 'down'):
                self.stop = vehicles[direction][lane][self.index - 1].stop
                - vehicles[direction][lane][self.index - 1].image.get_rect().height
                - stoppingGap
            elif (direction == 'up'):
                self.stop = vehicles[direction][lane][self.index - 1].stop
                + vehicles[direction][lane][self.index - 1].image.get_rect().height
                + stoppingGap
        else:
            self.stop = defaultStop[direction]

        # Set new starting and stopping coordinate
        if (direction == 'right'):
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] -= temp
        elif (direction == 'left'):
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] += temp
        elif (direction == 'down'):
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] -= temp
        elif (direction == 'up'):
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        if self.direction not in vehiclesNotTurned:
            vehiclesNotTurned[self.direction] = {}
        if self.lane not in vehiclesNotTurned[self.direction]:
            vehiclesNotTurned[self.direction][self.lane] = []

        if simulation_time < initial_red_time:
            if (self.direction == 'right' and self.x + self.image.get_rect().width >= stopLines[self.direction]) or \
                    (self.direction == 'down' and self.y + self.image.get_rect().height >= stopLines[self.direction]) or \
                    (self.direction == 'left' and self.x <= stopLines[self.direction]) or \
                    (self.direction == 'up' and self.y <= stopLines[self.direction]):
                # Stop the vehicle by not updating its position
                return

        if (self.direction == 'right'):
            if (self.crossed == 0 and self.x + self.image.get_rect().width > stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if (self.willTurn == 0):
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1
            if (self.willTurn == 1):
                if (self.lane == 1):
                    if (self.crossed == 0 or self.x + self.image.get_rect().width < stopLines[self.direction] + 40):
                        if ((self.x + self.image.get_rect().width <= self.stop or (
                                currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.x + self.image.get_rect().width < (
                                vehicles[self.direction][self.lane][self.index - 1].x - movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.x += self.speed
                    else:
                        if (self.turned == 0):
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x += 2.4
                            self.y -= 2.8
                            if (self.rotateAngle == 90):
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or (self.y > (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y +
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1].image.get_rect().height + movingGap))):
                                self.y -= self.speed
                elif (self.lane == 2):
                    if (self.crossed == 0 or self.x + self.image.get_rect().width < mid[self.direction]['x']):
                        if ((self.x + self.image.get_rect().width <= self.stop or (
                                currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.x + self.image.get_rect().width < (
                                vehicles[self.direction][self.lane][self.index - 1].x - movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.x += self.speed
                    else:
                        if (self.turned == 0):
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x += 2
                            self.y += 1.8
                            if (self.rotateAngle == 90):
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or ((self.y + self.image.get_rect().height) < (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y - movingGap))):
                                self.y += self.speed
            else:
                if (self.crossed == 0):
                    if ((self.x + self.image.get_rect().width <= self.stop or (
                            currentGreen == 0 and currentYellow == 0)) and (
                            self.index == 0 or self.x + self.image.get_rect().width < (
                            vehicles[self.direction][self.lane][self.index - 1].x - movingGap))):
                        self.x += self.speed
                else:
                    if ((self.crossedIndex == 0) or (self.x + self.image.get_rect().width < (
                            vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].x - movingGap))):
                        self.x += self.speed
        elif (self.direction == 'down'):
            if (self.crossed == 0 and self.y + self.image.get_rect().height > stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if (self.willTurn == 0):
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1
            if (self.willTurn == 1):
                if (self.lane == 1):
                    if (self.crossed == 0 or self.y + self.image.get_rect().height < stopLines[self.direction] + 50):
                        if ((self.y + self.image.get_rect().height <= self.stop or (
                                currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.y + self.image.get_rect().height < (
                                vehicles[self.direction][self.lane][self.index - 1].y - movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.y += self.speed
                    else:
                        if (self.turned == 0):
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x += 1.2
                            self.y += 1.8
                            if (self.rotateAngle == 90):
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or ((self.x + self.image.get_rect().width) < (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x - movingGap))):
                                self.x += self.speed
                elif (self.lane == 2):
                    if (self.crossed == 0 or self.y + self.image.get_rect().height < mid[self.direction]['y']):
                        if ((self.y + self.image.get_rect().height <= self.stop or (
                                currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.y + self.image.get_rect().height < (
                                vehicles[self.direction][self.lane][self.index - 1].y - movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.y += self.speed
                    else:
                        if (self.turned == 0):
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x -= 2.5
                            self.y += 2
                            if (self.rotateAngle == 90):
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or (self.x > (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x +
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1].image.get_rect().width + movingGap))):
                                self.x -= self.speed
            else:
                if (self.crossed == 0):
                    if ((self.y + self.image.get_rect().height <= self.stop or (
                            currentGreen == 1 and currentYellow == 0)) and (
                            self.index == 0 or self.y + self.image.get_rect().height < (
                            vehicles[self.direction][self.lane][self.index - 1].y - movingGap))):
                        self.y += self.speed
                else:
                    if ((self.crossedIndex == 0) or (self.y + self.image.get_rect().height < (
                            vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].y - movingGap))):
                        self.y += self.speed
        elif (self.direction == 'left'):
            if (self.crossed == 0 and self.x < stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if (self.willTurn == 0):
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1
            if (self.willTurn == 1):
                if (self.lane == 1):
                    if (self.crossed == 0 or self.x > stopLines[self.direction] - 70):
                        if ((self.x >= self.stop or (
                                currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x +
                                                             vehicles[self.direction][self.lane][
                                                                 self.index - 1].image.get_rect().width + movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.x -= self.speed
                    else:
                        if (self.turned == 0):
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x -= 1
                            self.y += 1.2
                            if (self.rotateAngle == 90):
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or ((self.y + self.image.get_rect().height) < (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y - movingGap))):
                                self.y += self.speed
                elif (self.lane == 2):
                    if (self.crossed == 0 or self.x > mid[self.direction]['x']):
                        if ((self.x >= self.stop or (
                                currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x +
                                                             vehicles[self.direction][self.lane][
                                                                 self.index - 1].image.get_rect().width + movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.x -= self.speed
                    else:
                        if (self.turned == 0):
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x -= 1.8
                            self.y -= 2.5
                            if (self.rotateAngle == 90):
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or (self.y > (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y +
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1].image.get_rect().height + movingGap))):
                                self.y -= self.speed
            else:
                if (self.crossed == 0):
                    if ((self.x >= self.stop or (currentGreen == 2 and currentYellow == 0)) and (
                            self.index == 0 or self.x > (
                            vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][
                        self.index - 1].image.get_rect().width + movingGap))):
                        self.x -= self.speed
                else:
                    if ((self.crossedIndex == 0) or (self.x > (
                            vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].x +
                            vehiclesNotTurned[self.direction][self.lane][
                                self.crossedIndex - 1].image.get_rect().width + movingGap))):
                        self.x -= self.speed
        elif (self.direction == 'up'):
            if (self.crossed == 0 and self.y < stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if (self.willTurn == 0):
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1
            if (self.willTurn == 1):
                if (self.lane == 1):
                    if (self.crossed == 0 or self.y > stopLines[self.direction] - 60):
                        if ((self.y >= self.stop or (
                                currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y +
                                                             vehicles[self.direction][self.lane][
                                                                 self.index - 1].image.get_rect().height + movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.y -= self.speed
                    else:
                        if (self.turned == 0):
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x -= 2
                            self.y -= 1.2
                            if (self.rotateAngle == 90):
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or (self.x > (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x +
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1].image.get_rect().width + movingGap))):
                                self.x -= self.speed
                elif (self.lane == 2):
                    if (self.crossed == 0 or self.y > mid[self.direction]['y']):
                        if ((self.y >= self.stop or (
                                currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y +
                                                             vehicles[self.direction][self.lane][
                                                                 self.index - 1].image.get_rect().height + movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.y -= self.speed
                    else:
                        if (self.turned == 0):
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x += 1
                            self.y -= 1
                            if (self.rotateAngle == 90):
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or (self.x < (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x -
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1].image.get_rect().width - movingGap))):
                                self.x += self.speed
            else:
                if (self.crossed == 0):
                    if ((self.y >= self.stop or (currentGreen == 3 and currentYellow == 0)) and (
                            self.index == 0 or self.y > (
                            vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][
                        self.index - 1].image.get_rect().height + movingGap))):
                        self.y -= self.speed
                else:
                    if ((self.crossedIndex == 0) or (self.y > (
                            vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].y +
                            vehiclesNotTurned[self.direction][self.lane][
                                self.crossedIndex - 1].image.get_rect().height + movingGap))):
                        self.y -= self.speed

                    # Initialization of signals with default values


lanes = [[], [], [], []]  # Assuming 4 lanes


def initialize():
    # Calculate green times, yellow time, and red time based on traffic lengths
    green_times, yellow_time, red_time = calculate_signal_timing()

    if (randomGreenSignalTimer):
        ts1 = TrafficSignal(5, 2, 5 + green_times[0])
        signals.append(ts1)
        ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, yellow_time, green_times[1])
        signals.append(ts2)
        ts3 = TrafficSignal(red_time, yellow_time, green_times[2])
        signals.append(ts3)
        ts4 = TrafficSignal(red_time, yellow_time, green_times[3])
        signals.append(ts4)
    else:
        ts1 = TrafficSignal(0, defaultYellow, green_times[0])
        signals.append(ts1)
        ts2 = TrafficSignal(ts1.yellow + ts1.green, defaultYellow, green_times[1])
        signals.append(ts2)
        ts3 = TrafficSignal(defaultRed, defaultYellow, green_times[2])
        signals.append(ts3)
        ts4 = TrafficSignal(defaultRed, defaultYellow, green_times[3])
        signals.append(ts4)

    repeat()


# Array containing traffic length for each direction
traffic_lengths = [60, 50, 30, 40]  # Example values for each direction (right, down, left, up)


def calculate_signal_timing():
    total_traffic_length = sum(traffic_lengths)  # Total length of traffic in all directions

    green_times = []
    for traffic_length in traffic_lengths:
        green_time = (traffic_length / total_traffic_length) * 24
        green_times.append(int(green_time))

    yellow_time = int(sum(green_times) * 0.2 / len(green_times))  # 20% of average green time as yellow
    red_time = defaultRed  # Keep red time as default

    return green_times, yellow_time, red_time


def repeat():
    global currentGreen, currentYellow, nextGreen
    while (signals[currentGreen].green > 0):  # while the timer of current green signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 1  # set yellow signal on
    # reset stop coordinates of lanes and vehicles
    for i in range(0, 3):
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while (signals[currentGreen].yellow > 0):  # while the timer of current yellow signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 0  # set yellow signal off

    # Calculate new timings based on the formula
    green_times, yellow_time, red_time = calculate_signal_timing()

    # Apply the new timings
    signals[currentGreen].green = green_times[currentGreen]
    signals[currentGreen].yellow = yellow_time
    signals[currentGreen].red = red_time

    currentGreen = nextGreen  # set next signal as green signal
    nextGreen = (currentGreen + 1) % noOfSignals  # set next green signal
    signals[nextGreen].red = signals[currentGreen].yellow + signals[
        currentGreen].green  # set the red time of next to next signal as (yellow time + green time) of next signal
    repeat()


# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if (i == currentGreen):
            if (currentYellow == 0):
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


# Generating vehicles in the simulation
def generateVehicles():
    while (True):
        vehicle_type = random.choice(allowedVehicleTypesList)
        lane_number = random.randint(1, 2)
        will_turn = 0
        if (lane_number == 1):
            temp = random.randint(0, 99)
            if (temp < 40):
                will_turn = 1
        elif (lane_number == 2):
            temp = random.randint(0, 99)
            if (temp < 40):
                will_turn = 1
        temp = random.randint(0, 99)
        direction_number = 0
        dist = [25, 50, 75, 100]
        if (temp < dist[0]):
            direction_number = 0
        elif (temp < dist[1]):
            direction_number = 1
        elif (temp < dist[2]):
            direction_number = 2
        elif (temp < dist[3]):
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number],
                will_turn)
        time.sleep(1)


def initialize_random_vehicles():
    stoppingGap = 40  # Distance between vehicles
    lanes_per_direction = 3  # Number of lanes per direction

    # Define the fixed number of vehicles for each direction
    vehicles_per_direction = {
        'right': 3,
        'down': 5,
        'left': 3,
        'up': 1
    }

    for direction in ['right', 'down', 'left', 'up']:
        num_vehicles = vehicles_per_direction[direction]  # Get the fixed number of vehicles for this direction

        for i in range(num_vehicles):
            vehicleClass = random.choice(['car', 'bus', 'truck'])  # Randomly select vehicle class
            will_turn = random.choice([0, 1])  # Randomly decide if the vehicle will turn
            lane = random.randint(1, lanes_per_direction - 1)  # Randomly select one of the 3 lanes

            vehicle = Vehicle(lane, vehicleClass, 0, direction, will_turn)

            # Adjust the vehicle's position based on the stop line and stack them backward with a gap
            if direction == 'right':
                vehicle.x = defaultStop[direction] - (i + 1) * (
                        40 + stoppingGap)  # Stack backward by 40 pixels plus gap
            elif direction == 'down':
                vehicle.y = defaultStop[direction] - (i + 1) * (
                        40 + stoppingGap)  # Stack downward by 40 pixels plus gap
            elif direction == 'left':
                vehicle.x = defaultStop[direction] + (i + 1) * (40 + stoppingGap)  # Stack forward by 40 pixels plus gap
            elif direction == 'up':
                vehicle.y = defaultStop[direction] + (i + 1) * (40 + stoppingGap)
                # Stack upward by 40 pixels plus gap

            vehicle.crossed = 0


class Main:
    global allowedVehicleTypesList
    i = 0
    for vehicleType in allowedVehicleTypes:
        if (allowedVehicleTypes[vehicleType]):
            allowedVehicleTypesList.append(i)
        i += 1
    thread1 = threading.Thread(name="initialization", target=initialize, args=())  # initialization
    thread1.daemon = True
    thread1.start()

    # Colours
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Screensize
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Setting background image i.e. image of intersection
    background = pygame.image.load(
        'D:\\python project\\car project\\Adaptive-Traffic-Signal-Timer-main\\Adaptive-Traffic-Signal-Timer-main\\Code\\YOLO\\darkflow\\images\\mod_int.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Loading signal images and font
    redSignal = pygame.image.load(
        'D:\\python project\\car project\\Adaptive-Traffic-Signal-Timer-main\\Adaptive-Traffic-Signal-Timer-main\\Code\\YOLO\\darkflow\\images\\signals\\red.png')
    yellowSignal = pygame.image.load(
        'D:\\python project\\car project\\Adaptive-Traffic-Signal-Timer-main\\Adaptive-Traffic-Signal-Timer-main\\Code\\YOLO\\darkflow\\images\\signals\\yellow.png')
    greenSignal = pygame.image.load(
        'D:\\python project\\car project\\Adaptive-Traffic-Signal-Timer-main\\Adaptive-Traffic-Signal-Timer-main\\Code\\YOLO\\darkflow\\images\\signals\\green.png')
    font = pygame.font.Font(None, 30)

    # thread2 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())  # Generating vehicles
    # thread2.daemon = True
    # thread2.start()

    initialize_random_vehicles()
    initial_red_time = 5  # 5 seconds for initial red signals
    global simulation_time
    simulation_time = 0  # Initialize simulation time
    clock = pygame.time.Clock()
    while True:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds (for 60 FPS)
        simulation_time += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        screen.blit(background, (0, 0))
        if simulation_time < initial_red_time:
            # Set all signals to red during the initial red phase
            for i in range(0, noOfSignals):
                screen.blit(redSignal, signalCoods[i])
                signals[i].signalText = signals[i].red

        else:
            for i in range(0,
                           noOfSignals):  # display signal and set timer according to current status: green, yello, or red
                if (i == currentGreen):
                    if (currentYellow == 1):
                        signals[i].signalText = signals[i].yellow
                        screen.blit(yellowSignal, signalCoods[i])
                    else:
                        signals[i].signalText = signals[i].green
                        screen.blit(greenSignal, signalCoods[i])
                else:
                    if (signals[i].red <= 10):
                        signals[i].signalText = signals[i].red
                    else:
                        signals[i].signalText = "---"
                    screen.blit(redSignal, signalCoods[i])
        signalTexts = ["", "", "", ""]

        # display signal timer
        for i in range(0, noOfSignals):
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])

        # display the vehicles
        for vehicle in simulation:
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()

        highlight_density_area(simulation, screen)
        pygame.display.update()


Main()