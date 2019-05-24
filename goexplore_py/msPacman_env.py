# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from .import_ai import *
from baselines.common.atari_wrappers import *
from diverseExplorer import MyEpisodicLifeEnv

class PacmanPosLevel:
    __slots__ = ['level', 'score' 'room', 'x', 'y', 'tuple']

    def __init__(self, level, x, y):
        self.level = level
        self.room = level
        self.score = None
        self.x = x
        self.y = y


        self.set_tuple()

    def set_tuple(self):
        self.tuple = (self.level, self.x, self.y)

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        if not isinstance(other, PacmanPosLevel):
            return False
        return self.tuple == other.tuple

    def __getstate__(self):
        return self.tuple

    def __setstate__(self, d):
        self.level, self.x, self.y = d
        self.tuple = d

    def __repr__(self):
        return f'Level={self.level} x={self.x} y={self.y}'


def convert_state(state):
    import cv2
    return ((cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), MyMsPacman.TARGET_SHAPE, interpolation=cv2.INTER_AREA) / 255.0) * MyMsPacman.MAX_PIX_VALUE).astype(np.uint8)


PYRAMID = [
    [-1, -1, -1, 0, 1, 2, -1, -1, -1],
    [-1, -1, 3, 4, 5, 6, 7, -1, -1],
    [-1, 8, 9, 10, 11, 12, 13, 14, -1],
    [15, 16, 17, 18, 19, 20, 21, 22, 23]
]

OBJECT_PIXELS = [
    50,  # Hammer/mallet
    40,  # Key 1
    40,  # Key 2
    40,  # Key 3
    37,  # Sword 1
    37,  # Sword 2
    42   # Torch
]

KNOWN_XY = [None] * 24

KEY_BITS = 0x8 | 0x4 | 0x2


def get_room_xy(room):
    if KNOWN_XY[room] is None:
        for y, l in enumerate(PYRAMID):
            if room in l:
                KNOWN_XY[room] = (l.index(room), y)
                break
    return KNOWN_XY[room]


def clip(a, m, M):
    if a < m:
        return m
    if a > M:
        return M
    return a


class MyMsPacman:
    def __init__(self, check_death: bool = True, unprocessed_state: bool = False,
                 x_repeat=2, ):  # TODO: version that also considers the room objects were found in
        self.env = FrameStack(WarpFrame(MyEpisodicLifeEnv(gym.make('MsPacmanNoFrameskip-v4'))),4)
        self.env.reset()

        self.ram = None
        self.check_death = check_death
        self.cur_steps = 0
        self.cur_score = 0
        self.rooms = {}
        self.room_time = None
        self.room_threshold = 40
        self.idle_on_new_level = 260
        self.unwrapped.seed(0)
        self.unprocessed_state = unprocessed_state
        self.state = []
        self.ram_death_state = -1
        self.x_repeat = x_repeat
        self.cur_lives = 5
        self.ignore_ram_death = False


    def __getattr__(self, e):
        return getattr(self.env, e)

    def reset(self) -> np.ndarray:
        observation= self.env.reset()
        for _ in range(self.idle_on_new_level):
            self.env.step(0)
        unprocessed_state = self.env.unwrapped._get_obs()
        self.cur_lives = 3
        self.state = [convert_state(unprocessed_state)]
        for _ in range(3):
            observation = self.env.step(0)[0]
            unprocessed_state = self.env.unwrapped._get_obs()
            self.state.append(convert_state(unprocessed_state))
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = 0
        self.cur_steps = 0
        self.ram_death_state = -1
        self.pos = None
        self.pos = self.pos_from_unprocessed_state(self.get_face_pixels(unprocessed_state), unprocessed_state)
        if self.get_pos().level not in self.rooms:
            self.rooms[self.get_pos().level] = (False, unprocessed_state[50:].repeat(self.x_repeat, axis=1))
        self.room_time = (self.get_pos().level, 0)
        if self.unprocessed_state:
            return observation

        return copy.copy(self.state)

    def pos_from_unprocessed_state(self, face_pixels, unprocessed_state):
        face_pixels = [(y, x * self.x_repeat) for y, x in face_pixels]
        if len(face_pixels) == 0:
            assert self.pos != None, 'No face pixel and no previous pos'
            return self.pos  # Simply re-use the same position
        y, x = np.mean(face_pixels, axis=0)
        level = 0

        if self.pos is not None:
            level = self.pos.level

            direction_x = clip(int((self.pos.x - x) / 10), -1, 1)
            direction_y = clip(int((self.pos.y - y) / 10), -1, 1)
            if direction_x != 0 or direction_y != 0:
                if y > 100 and y > 110 and x > 150 and x < 170:
                    level += 1



        return PacmanPosLevel(level, x, y)



    def get_restore(self):
        return (
            self.unwrapped.clone_full_state(),
            copy.copy(self.state),
            self.cur_score,
            self.cur_steps,
            self.pos,
            self.room_time,
            self.ram_death_state,

            self.cur_lives
        )

    def restore(self, data):
        (full_state, state, score, steps, pos, room_time, ram_death_state, self.cur_lives) = data
        self.state = copy.copy(state)
        self.env.reset()
        self.unwrapped.restore_full_state(full_state)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_score = score
        self.cur_steps = steps
        self.pos = pos
        self.room_time = room_time
        self.ram_death_state = ram_death_state

        if self.unprocessed_state:
            self.env.step(0) # take noop to update screen, this is not optimal solution
            obs = self.env.env.observation(self.env.unwrapped._get_obs())
            for _ in range(self.env.k):
                self.env.frames.append(obs) # fill the frame stack
            obs = self.env._get_ob() #get new obs
        else:
            obs = copy.copy(self.state)
        return obs

    def is_transition_screen(self, unprocessed_state):
        unprocessed_state = unprocessed_state[50:, :, :]
        # The screen is a transition screen if it is all black or if its color is made up only of black and
        # (0, 28, 136), which is a color seen in the transition screens between two levels.
        return (
                       np.sum(unprocessed_state[:, :, 0] == 0) +
                       np.sum((unprocessed_state[:, :, 1] == 0) | (unprocessed_state[:, :, 1] == 28)) +
                       np.sum((unprocessed_state[:, :, 2] == 0) | (unprocessed_state[:, :, 2] == 136))
               ) == unprocessed_state.size

    def get_face_pixels(self, unprocessed_state):
        # TODO: double check that this color does not re-occur somewhere else
        # in the environment.
        return set(zip(*np.where(unprocessed_state[:-40, :,0] == 210)))

    def is_pixel_death(self, unprocessed_state, face_pixels):
        # There are no face pixels and yet we are not in a transition screen. We
        # must be dead!
        if len(face_pixels) == 0:
            # All of the screen except the bottom is black: this is not a death but a
            # room transition. Ignore.
            if self.is_transition_screen(unprocessed_state):
                return False
            return True

        # We already checked for the presence of no face pixels, however,
        # sometimes we can die and still have face pixels. In those cases,
        # the face pixels will be DISCONNECTED.
        for pixel in face_pixels:
            for neighbor in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (pixel[0] + neighbor[0], pixel[1] + neighbor[1]) in face_pixels:
                    return False

        return True

    def is_ram_death(self):
        if self.ram[58] > self.cur_lives:
            self.cur_lives = self.ram[58]
        return self.ram[55] != 0 or self.ram[58] < self.cur_lives

    def step(self, action) -> typing.Tuple[np.ndarray, float, bool, dict]:
        observation, reward, done, lol = self.env.step(action)
        unprocessed_state = self.env.unwrapped._get_obs()
        self.state.append(convert_state(unprocessed_state))
        self.state.pop(0)
        self.ram = self.env.unwrapped.ale.getRAM()
        self.cur_steps += 1

        face_pixels = self.get_face_pixels(unprocessed_state)


        self.cur_score += reward
        self.pos = self.pos_from_unprocessed_state(face_pixels, unprocessed_state)
        if self.pos.level != self.room_time[0]:
            for _ in range(self.idle_on_new_level):
                self.env.step(0)
            self.room_time = (self.pos.level, 0)

        self.room_time = (self.pos.room, self.room_time[1] + 1)
        if (self.pos.level not in self.rooms or
                (self.room_time[1] == self.room_threshold and
                 not self.rooms[self.pos.level][0])):
            self.rooms[self.pos.level] = (
                self.room_time[1] == self.room_threshold,
                unprocessed_state[:-40].repeat(self.x_repeat, axis=1)
            )
        if self.unprocessed_state:
            return observation, reward, done, lol
        return copy.copy(self.state), reward, done, lol

    def get_pos(self):
        assert self.pos is not None
        return self.pos

    def render_with_known(self, known_positions, resolution, show=True, filename=None, combine_val=max,
                          get_val=lambda x: x.score, minmax=None):
        height, width = list(self.rooms.values())[0][1].shape[:2]

        final_image = np.zeros((height * 4, width * 9, 3), dtype=np.uint8) + 255

        positions = PYRAMID

        def room_pos(room):
            for height, l in enumerate(positions):
                for width, r in enumerate(l):
                    if r == room:
                        return (height, width)
            return None

        points = defaultdict(int)

        for level in range(24):
            if level in self.rooms:
                img = self.rooms[level][1]
            else:
                img = np.zeros((height, width, 3)) + 127
            y_room, x_room = room_pos(level)
            y_room *= height
            x_room *= width
            final_image[y_room:y_room + height, x_room:x_room + width, :] = img

        plt.figure(figsize=(final_image.shape[1] // 30, final_image.shape[0] // 30))

        for room in range(24):
            y_room, x_room = room_pos(room)
            y_room *= height
            x_room *= width

            for i in np.arange(resolution, img.shape[0], resolution):
                cv2.line(final_image, (x_room, y_room + i), (x_room + img.shape[1], y_room + i), (127, 127, 127), 1)
                # plt.plot([x_room, x_room + img.shape[1]], [y_room + i, y_room + i], '--', linewidth=1, color='gray')
            for i in np.arange(resolution, img.shape[1], resolution):
                cv2.line(final_image, (x_room + i, y_room), (x_room + i, y_room + img.shape[0]), (127, 127, 127), 1)
                # plt.plot([x_room + i, x_room + i], [y_room, y_room + img.shape[0]], '--', linewidth=1, color='gray')

            cv2.line(final_image, (x_room, y_room), (x_room, y_room + img.shape[0]), (255, 255, 255), 1)
            cv2.line(final_image, (x_room, y_room), (x_room + img.shape[1], y_room), (255, 255, 255), 1)
            cv2.line(final_image, (x_room + img.shape[1], y_room), (x_room + img.shape[1], y_room + img.shape[0]),
                     (255, 255, 255), 1)
            cv2.line(final_image, (x_room, y_room + img.shape[0]), (x_room + img.shape[1], y_room + img.shape[0]),
                     (255, 255, 255), 1)

            for k in known_positions:
                if k.room != room:
                    continue
                x = x_room + (k.x * resolution + resolution / 2)
                y = y_room + (k.y * resolution + resolution / 2)
                points[(x, y)] = combine_val(points[(x, y)], get_val(k))

        plt.imshow(final_image)
        if minmax:
            points[(0, 0)] = minmax[0]
            points[(0, 1)] = minmax[1]

        vals = list(points.values())
        points = list(points.items())
        plt.scatter([p[0][0] for p in points], [p[0][1] for p in points], c=[p[1] for p in points], cmap='bwr',
                    s=(resolution) ** 2, marker='*')
        plt.legend()

        import matplotlib.cm
        import matplotlib.colors
        mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=np.min(vals), vmax=np.max(vals)),
                                                cmap='bwr')
        mappable.set_array(vals)
        matplotlib.rcParams.update({'font.size': 22})
        plt.colorbar(mappable)

        plt.axis('off')
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def get_room_xy(room):
        if KNOWN_XY[room] is None:
            for y, l in enumerate(PYRAMID):
                if room in l:
                    KNOWN_XY[room] = (l.index(room), y)
                    break
        return KNOWN_XY[room]

    @staticmethod
    def get_room_out_of_bounds(room_x, room_y):
        return room_y < 0 or room_x < 0 or room_y >= len(PYRAMID) or room_x >= len(PYRAMID[0])

    @staticmethod
    def get_room_from_xy(room_x, room_y):
        return PYRAMID[room_y][room_x]

    @staticmethod
    def make_pos(score, pos):
        return PacmanPosLevel(pos.level, pos.x, pos.y)
