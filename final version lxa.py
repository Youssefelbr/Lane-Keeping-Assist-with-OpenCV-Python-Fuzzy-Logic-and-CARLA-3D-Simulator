

from __future__ import print_function
import matplotlib.pyplot as plt
import time
import keyboard
import glob
import os
import sys
import skfuzzy as fuzz
from skfuzzy import control as ctrl
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla

from carla import ColorConverter as cc
import cv2
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_4
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

class World(object):
    def __init__(self, carla_world, hud, actor_filter, actor_role_name='hero'):
        self.world = carla_world
        self.actor_role_name = actor_role_name
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None

        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0



        # Get a random blueprint.
        # blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        tesla_blueprint_name = "vehicle.tesla.model3"
        blueprint = self.world.get_blueprint_library().find(tesla_blueprint_name)
        # pedestrian_bp1 = self.world.get_blueprint_library().filter('walker.pedestrian.*')[0]
        # spawn_point_1 = carla.Transform()
        # spawn_point_1.location.x = 20  # Replace <x_coordinate> with the desired x-coordinate
        # spawn_point_1.location.y = 100  # Replace <y_coordinate> with the desired y-coordinate
        # spawn_point_1.location.z = 0





        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            ###"
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        # self._weather_index %= len(self._weather_presets)
        # preset = self._weather_presets[self._weather_index]
        # self.hud.notification('Weather: %s' % preset[1])
        # self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        # self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()
class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._s_key_pressed = False
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0

    def set_steer_value(self, steer):
        self._control.steer = steer

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_s]:              #when we press the s key and arrows it works manually
        #     if not self._s_key_pressed:  # Si la touche 'S' n'a pas été enfoncée précédemment
        #         self._s_key_pressed = True
        #         self._steer_cache = 0.0
        #     else:  # Si la touche 'S' a déjà été enfoncée précédemment
        #         self._s_key_pressed = False
        #         self._control.steer = 0.0  # Réinitialiser la direction lorsque la touche 'S' est relâchée
        # if self._s_key_pressed :
        #     self._steer_cache = 0.0
        #     self._control.steer = 0.0
            print("manual")
            if keys[K_RIGHT] :
                self._steer_cache -= steer_increment
            elif keys[K_LEFT] :
                self._steer_cache += steer_increment
            else :
                self._steer_cache = 0
            self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
            self._control.steer = round(self._steer_cache, 1)
        else:
            self._steer_cache = 0



        #print("steering angle ", self._control.steer)#display of the current steering value
        self._control.brake = 1.0 if keys[K_DOWN] else 0.0
        self._control.hand_brake = keys[K_SPACE]




    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.speedox = 0
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 10)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[1]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 20)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):

        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        self.speedox = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        # print(self.speedox)

        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        vehicles = world.world.get_actors().filter('vehicle.*')
        #print("les coordonnées sont ", t.location.x, t.location.y)#############################################"



        self._info_text = []
        if isinstance(c, carla.VehicleControl):
            self._info_text += []
        elif isinstance(c, carla.WalkerControl):
            self._info_text += []

        # if len(vehicles) > 1:
        #     self._info_text += ['Nearby vehicles:']
        #     distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
        #     vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
        #     for d, vehicle in sorted(vehicles):
        #         if d > 200.0:
        #             break
        #         vehicle_type = get_actor_display_name(vehicle, truncate=22)
        #         self._info_text.append('% 4dm %s' % (d, vehicle_type))
    def get_speed(self):
        return self.speedox

    def toggle_info(self):
        self._show_info = not self._show_info



    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            # for item in self._info_text:
            #     if v_offset + 18 > self.dim[1]:
            #         break
            #     if isinstance(item, list):
            #         if len(item) > 1:
            #             points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
            #             pygame.draw.lines(display, (255, 136, 0), False, points, 2)
            #         item = None
            #         v_offset += 18
            #     elif isinstance(item, tuple):
            #         if isinstance(item[1], bool):
            #             rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
            #             pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
            #         else:
            #             rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
            #             pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
            #             f = (item[1] - item[2]) / (item[3] - item[2])
            #             if item[2] < 0.0:
            #                 rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
            #             else:
            #                 rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
            #             pygame.draw.rect(display, (255, 255, 255), rect)
            #         item = item[0]
            #     if item:  # At this point has to be a str.
            #         surface = self._font_mono.render(item, True, (255, 255, 255))
            #         display.blit(surface, (8, v_offset))
            #     v_offset += 18
            #
            #


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.prev_lines = None
        self.xmiddle = 0
        self.erreur = 0
        self.un2 = 0
        self.previous_middle_line = None  # Store the coordinates of the previous middle line
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def get_erreur(self):
        return self.erreur ###########################a getter to get the error att value


    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))
    ####################################################################################lane detection function definition
    def region_of_interest(self, image):
        bl = (13, 599)
        bl1 = (12, 500)
        tl = (259, 386)
        tr = (654, 396)
        br1 = (813, 475)
        br = (865, 595)
        polygons  = np.array([[bl,bl1,tl,tr,br1,br]])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, (255,255,255))
        masked_image= cv2.bitwise_and(image,mask)
        return masked_image

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        #cv2.imshow('blur', thresh)
        canny = cv2.Canny(thresh, 90, 150)#150 prev set
        return canny

    def make_coordinates(self, image, line_parameters):
        try:
            slope, intercept = line_parameters
        except TypeError:
            slope, intercept = 0.01, 0.01
        y1 = image.shape[0]
        y2 = np.float64(y1 * (3.5 / 5))

        x1 = np.float64((y1 - intercept) / slope)
        x2 = np.float64((y2 - intercept) / slope)

        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self, image, lines):
        left_fit = []  # liste vide pour contenir les ligne de gauche
        right_fit = []  # liste vide pour contenir les ligne de droite
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                intercept = parameters[1]
                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
                left_fit_average = np.average(left_fit, axis=0)
                right_fit_average = np.average(right_fit, axis=0)
                left_line = self.make_coordinates(image, left_fit_average)
                right_line = self.make_coordinates(image, right_fit_average)

            if left_fit and right_fit :
                                            # Calculate middle line coordinates
                if left_line is not None:
                    if right_line is not None:
                        x1L, y1L, x2L, y2L = left_line
                        x1R, y1R, x2R, y2R = right_line

                        middle_line = np.array([(x1L + x1R) // 2, (y1L + y1R) // 2, (x2L + x2R) // 2, (y2L + y2R) // 2])#calcul du centre de la voie
                        if self.previous_middle_line is not None:#si on a cumulé des middle line
                            middle_line = np.array(middle_line)#on transforme en np array
                            self.previous_middle_line = np.array(self.previous_middle_line)#on transforme en np array
                            # Smooth the middle line coordinates using moving average
                            smoothed_middle_line = 0.2 * middle_line + (1 - 0.2) * self.previous_middle_line #application du smoothing
                        else:
                            smoothed_middle_line = middle_line#si on n'a rien cumulé , on affiche le current one

                        self.previous_middle_line = smoothed_middle_line #sauvegarde du dernier middle line
                        if len(smoothed_middle_line) == 4:
                            x1m = np.int32(smoothed_middle_line[0])
                            y1m = np.int32(smoothed_middle_line[1])
                            x2m = np.int32(smoothed_middle_line[2])
                            y2m = np.int32(smoothed_middle_line[3])
                            cv2.line(image, (x1m, y1m), (x2m, y2m), (255, 255, 0), 4)
                            cv2.line(image,(x2m,y1m),(x2m,y2m),(255,0,0),4)
                            return np.array([left_line, right_line]), np.array([x1m, y1m, x2m, y2m])

            return np.array([left_line, right_line]), np.array([0,0,0,0])

    def display_lines(self, image, lines,erreur):
        line_image = np.zeros_like(image)
        if isinstance(lines, np.ndarray) and lines.ndim > 0:
            if lines.ndim == 1:
                lines = lines.reshape(1, -1)
            for line in lines:
                if line is not None:
                    x1, y1, x2, y2 = line.reshape(4)
                    x1, y1, x2, y2 = np.int32(x1), np.int32(y1), np.int32(x2), np.int32(y2)
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
                    #affichage des lignes selon le sens du dépassement
                    if erreur > 5 :
                        if (x1-x2)> 5:#grandeur qui nous donne idée sur la ligne gauche ou droite
                            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 150), 10)
                    else :
                        if (x1-x2)< -5:
                            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 150), 10)
        return line_image

    def car_shape(self, image):
        height, width, _ = image.shape
        pts = np.array([[285, 600], [310,550], [585,550], [610, 600]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (255, 200, 201), 2)
        car_center = (285 + 610) // 2
        cv2.line(image, (car_center, 500), (car_center, int(height * 0.7)), (112, 200, 1),3)  # ////la lignes du milieu de la voiture
        return car_center

    def display_text_on_image(self,param, image):

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)

        # Get text size and position
        text1 = "Erreur de position laterale : " + str(param)

        text_size, _ = cv2.getTextSize(text1, font, 1, 2)

        cv2.putText(image, text1, (170, 150), font, 1, color, 2)



    def render(self, display):

        if self.surface is not None:
            # Convert the pygame surface to a numpy array
            img_rgb = pygame.surfarray.array3d(self.surface)
            # Convert the RGB image to BGR for OpenCV
            frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            resized_frame = cv2.resize(frame, (600,900))

            framers = cv2.rotate(resized_frame, cv2.ROTATE_90_CLOCKWISE)#fix to rotate image
            bl = (13, 599)
            bl1 = (12, 500)
            tl = (259, 386)
            tr = (654, 396)
            br1 = (813, 475)
            br = (865, 595)
            cv2.circle(framers, tl, 5, (0, 0, 255), -1)
            cv2.circle(framers, bl, 5, (0, 0, 255), -1)
            cv2.circle(framers, bl1, 5, (0, 0, 255), -1)
            cv2.circle(framers, tr, 5, (0, 0, 255), -1)
            cv2.circle(framers, br,5, (0, 0, 255), -1)
            cv2.circle(framers, br1, 5, (0, 0, 255), -1)


            canny = self.canny(framers)#applying all the filters canny , blur ,
            cropped_image = self.region_of_interest(canny)
            lines = cv2.HoughLinesP(cropped_image, 1, 0.01, 100, np.array([]), minLineLength= 30, maxLineGap=50)#applying the houghtransform function

            if self.prev_lines is not None:

                if lines is not None and len(lines) > 0:
                    smoothed_lines = 0.3 * self.average_slope_intercept(framers, lines)[0] + (1 - 0.3) * self.prev_lines
                else:
                    smoothed_lines = self.prev_lines
            else:
                if lines is not None and len(lines) > 0:
                    smoothed_lines = self.average_slope_intercept(framers, lines)[0]
                else:
                    smoothed_lines = None
            self.prev_lines = smoothed_lines
            #line_image = self.display_lines(framers, smoothed_lines,self.erreur)
            car_midline = self.car_shape(framers)
            if self.average_slope_intercept(framers, lines) is not None :
                self.xmiddle, un1, self.un2, un3 = self.average_slope_intercept(framers, lines)[1]
            self.erreur = car_midline - self.un2
            # print(self.erreur)
            #affichage graphique
            self.display_text_on_image(self.erreur,framers)


            #fin de l'affichage
            line_image = self.display_lines(framers, smoothed_lines,self.erreur)
            combo_image = cv2.addWeighted(framers, 0.8, line_image, 1, 1)


            # Display the image in an OpenCV window
            cv2.imshow('Camera Feed', combo_image)

            cv2.waitKey(1)

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.camera'):

            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


def fuzzy_controller(input_error, speed_car):
    if input_error < 150 and input_error > -150:
        # Definition des grandeurs floue d'entrées
        error = ctrl.Antecedent(np.arange(-150, 151, 1), 'error')
        speed = ctrl.Antecedent(np.arange(0, 61, 1), 'speed')
        # pour la sortie qui'est l'angle de correction
        correction = ctrl.Consequent(np.arange(-0.2, 0.21, 0.001), 'correction')

        # Définir les fonctions d'appartenance pour les ensembles floues
        # fonction triangulaire pour l'erreur
        error['very_Low_L'] = fuzz.trimf(error.universe,  [-150, -150, -64.36])
        error['Left'] = fuzz.trimf(error.universe,  [-99.5, -67.61, 0])
        error['Center'] = fuzz.trimf(error.universe,  [-49.1, 0.325, 49.73])
        error['right'] = fuzz.trimf(error.universe,   [0.325, 57.2, 100])
        error['very_high_R'] = fuzz.trimf(error.universe,   [64.03, 150, 150])


        # creation des fonction d'appartenance triangulaire pour la grandeur floue de sortie
        # pour l'angle de la correction
        correction['left'] = fuzz.trapmf(correction.universe,   [-0.2, -0.17, -0.1, -0.028])
        correction['slightly_left'] = fuzz.trimf(correction.universe,    [-0.05, -0.035, -0.011])
        correction['no_steering'] = fuzz.trimf(correction.universe,  [-0.02, 0, 0.02])
        correction['slightly_right'] = fuzz.trimf(correction.universe,  [0.011, 0.035, 0.05])
        correction['right'] = fuzz.trapmf(correction.universe,    [0.028, 0.1, 0.17, 0.2])
        # creation des fonction d'appartenance triangulaire pour la grandeur d'entree speed
        speed['slow'] = fuzz.trimf(speed.universe,  [0, 0, 12.59])
        speed['medium'] = fuzz.trimf(speed.universe,  [0.07711, 13.09, 26.59])
        speed['high'] = fuzz.trimf(speed.universe,  [17, 24.7, 35.4])
        speed['very_high'] = fuzz.trapmf(speed.universe,   [26.8, 39.9, 49.6, 50])


        # Définition des règles floues
        rules = [
            # ctrl.Rule(error['very_Low_L'] & speed['slow'], correction['slightly_right']),
            # ctrl.Rule(error['very_Low_L'] & speed['medium'], correction['slightly_right']),
            # ctrl.Rule(error['very_Low_L'] & speed['high'], correction['slightly_right']),
            # ctrl.Rule(error['very_Low_L'] & speed['very_high'], correction['slightly_right']),

            ctrl.Rule(error['Left'] & speed['slow'], correction['slightly_right']),
            ctrl.Rule(error['Left'] & speed['medium'], correction['slightly_right']),
            ctrl.Rule(error['Left'] & speed['high'], correction['slightly_right']),
            ctrl.Rule(error['Left'] & speed['very_high'], correction['slightly_right']),

            ctrl.Rule(error['right'] & speed['slow'], correction['slightly_left']),
            ctrl.Rule(error['right'] & speed['medium'], correction['slightly_left']),
            ctrl.Rule(error['right'] & speed['high'], correction['slightly_left']),
            ctrl.Rule(error['right'] & speed['very_high'], correction['slightly_left']),

            ctrl.Rule(error['Center'] & speed['slow'], correction['no_steering']),
            ctrl.Rule(error['Center'] & speed['medium'], correction['no_steering']),
            ctrl.Rule(error['Center'] & speed['high'], correction['no_steering']),
            ctrl.Rule(error['Center'] & speed['very_high'], correction['no_steering']),

            # ctrl.Rule(error['very_high_R'] & speed['slow'], correction['left']),
            # ctrl.Rule(error['very_high_R'] & speed['medium'], correction['slightly_left']),
            # ctrl.Rule(error['very_high_R'] & speed['high'], correction['slightly_left']),
            # ctrl.Rule(error['very_high_R'] & speed['very_high'], correction['slightly_left']),

        ]
        # Création du système de contrôle flou
        controller = ctrl.ControlSystem(rules)
        control_simulation = ctrl.ControlSystemSimulation(controller)

        # Évaluation du système de contrôle flou avec la valeur d'erreur donnée
        control_simulation.input['error'] = input_error
        control_simulation.input['speed'] = speed_car
        try:

            control_simulation.compute()
            output_correction = control_simulation.output['correction']
            print(output_correction)

        except:
            output_correction = 0
    else:
        output_correction = 0

    return output_correction

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter, args.rolename)
        controller = KeyboardControl(world, args.autopilot)
        Erreur = []  # Store the sp values
        Speed = []  # Store the sp values
        commande_values = []  # Store the sp values







        clock = pygame.time.Clock()
        last_time = time.time()
        recording = False  # Variable pour indiquer si l'enregistrement est en cours
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(12, 4))
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            current_time = time.time()

            #begining of the steering wheel autocorrection
            erreur = world.camera_manager.get_erreur()

            # print("lerreur est ",erreur)
            # #speed getting
            sp = int(world.hud.get_speed())
            #
            # print("la vitesse est : ",sp)
            #fuzzy_controller implementation:
        #     commande_volant = fuzzy_controller(erreur,sp)
        #     # print("steering wheel angle :",commande_volant)
        #     if erreur >= -5 and erreur<=5:
        #         controller.set_steer_value(erreur/9000)
        #     else:
        #         controller.set_steer_value(-commande_volant)
        #     deg = commande_volant/0.0028
        #     cm = erreur/2
        #
        #     if keyboard.is_pressed('enter'):
        #         if not recording:
        #             # Démarrer l'enregistrement
        #             recording = True
        #         else:
        #             # Arrêter l'enregistrement
        #             recording = False
        #     if recording:
        #         #premier affichage
        #
        #
        #         commande_values.append(deg)  # Store the sp value
        #         ax1.plot(commande_values, color='blue')
        #         ax1.set_title("Réponse du controlleur FLOUE")
        #         ax1.set_xlabel("temps (ms)")
        #         ax1.set_ylabel("commande volant (°)")
        #
        #         # plt.figure('Réponse du controlleur FLOUE')
        #         # plt.plot(speed,color ='blue')
        #         # plt.title("Réponse du controlleur FLOUE")
        #         # plt.xlabel("temps (ms)")
        #         # plt.ylabel("commande volant")
        #
        #         Erreur.append(cm)
        #         ax2.plot(Erreur, color='red')
        #         ax2.set_title("Erreur de la position latérale")
        #         ax2.set_xlabel("temps (ms)")
        #         ax2.set_ylabel("Erreur")
        #
        #         Speed.append(sp)
        #         ax3.plot(Speed, color='green')
        #         ax3.set_title("Vitesse instantanée ")
        #         ax3.set_xlabel("temps (ms)")
        #         ax3.set_ylabel("vitesse (km/h")
        #
        #
        #         # plt.figure('Erreur de la position latérale')
        #         # plt.plot(Erreur,color ='red')
        #         # plt.title("Erreur de la position latérale")
        #         # plt.xlabel("temps (ms)")
        #         # plt.ylabel("Erreur")
        #
        #         # plt.tight_layout()
        #
        #         plt.pause(0.000000001)
        # plt.show()
        #




    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()


        pygame.quit()
def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)
    try:

        game_loop(args)


    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
