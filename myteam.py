# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


# helper function
def to_int(pos):
    # convert potential float positions (x.0, y.0) to integer grid coords
    # avoids weird float-equality bugs when comparing positions later
    return (int(pos[0]), int(pos[1]))


#################
# Team creation #
#################


def create_team(
    first_index,
    second_index,
    is_red,
    first="OffensiveReflexAgent",
    second="DefensiveReflexAgent",
    num_training=0,
):
    """
    Simple factory to build two agents. Uses eval on the class name strings
    (this is fine for this local project but wouldn't be ideal in general).
    Note: is_red is provided by the framework but this function doesn't use
    it directly — team agents can check self.red inside their code.
    """
    # quick and dirty: instantiate classes by name (works because classes are in scope)
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height

        # middle of map used to decide "home" vs "enemy" side
        self.mid_x = self.width // 2
        self.mid_y = self.height // 2
        self.start = to_int(game_state.get_agent_position(self.index))

        # offensive agent extra states
        self.was_retreating = False  # previously retreating?
        self.retreat_cooldown = 0  # small cooldown to avoid bouncing between states
        self.last_dir = None  # remember last chosen direction to penalize reversing

        # more helper functions

    def is_home_side(self, pos):
        # if we're red, home is left half (x < mid). if blue, home is right half.
        # this matters for deciding whether to defend or attack.
        x, y = pos
        return x < self.mid_x if self.red else x >= self.mid_x

    def is_enemy_side(self, pos):
        return not self.is_home_side(pos)

    # A* with optional ghost penalty
    def astar(self, game_state, start, goals, avoid_ghosts=None):
        """
        start: (x,y) start position (floats allowed)
        goals: list of positions to reach
        avoid_ghosts: list of positions to add a small penalty around (discourages paths near ghosts)
        returns: list of positions (path excluding start) or None if no path found
        """
        start = to_int(start)
        goals = [to_int(g) for g in goals]
        if avoid_ghosts is None:
            avoid_ghosts = []

        walls = self.walls

        open_set = util.PriorityQueue()
        open_set.push(start, 0)

        came_from = {start: None}
        g_cost = {start: 0}
        closed = set()

        while not open_set.is_empty():
            current = open_set.pop()
            if current in closed:
                continue
            closed.add(current)

            if current in goals:
                # reconstruct path from start: goal, return positions excluding the start
                path = []
                while came_from[current] is not None:
                    path.append(current)
                    current = came_from[current]
                return list(reversed(path))

            x, y = current
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # basic boundaries and wall checks
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                if walls[nx][ny]:
                    continue

                nxt = (nx, ny)
                new_cost = g_cost[current] + 1

                # penalize being too close to ghosts
                for gpos in avoid_ghosts:
                    if util.manhattan_distance(nxt, gpos) <= 2:
                        new_cost += 6

                if nxt not in g_cost or new_cost < g_cost[nxt]:
                    g_cost[nxt] = new_cost
                    # heuristic is the manhattan distance to nearest goal
                    h = min(util.manhattan_distance(nxt, g) for g in goals)
                    came_from[nxt] = current
                    open_set.push(nxt, new_cost + h)

        return None  # no path found

    def evaluate(self, s, a):
        return self.get_features(s, a) * self.get_weights(s, a)


class OffensiveReflexAgent(ReflexCaptureAgent):

    # strategy is to take some unprotected food in top and bottom and retreat before dying and win with some points advantage

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = to_int(my_state.get_position())

        # cooldown reset: once we return home after retreating, set a short cooldown
        # so the agent doesn't immediately go back out and get killed again.
        if self.was_retreating and self.is_home_side(my_pos):
            self.retreat_cooldown = 15
            self.was_retreating = False

        # stay on our side and avoid reversing direction
        if self.retreat_cooldown > 0:
            self.retreat_cooldown -= 1
            legal = game_state.get_legal_actions(self.index)

            # prefer moves that keep us on home side while on cooldown
            safe = []
            for a in legal:
                succ = game_state.generate_successor(self.index, a)
                succ_state = succ.get_agent_state(self.index)
                succ_pos = to_int(succ_state.get_position())
                if self.is_home_side(succ_pos):
                    safe.append(a)

            if len(safe) > 0:
                actions = safe
            else:
                actions = legal

            # avoid reversing the previous move to reduce oscillation
            reverse_dir = {
                Directions.NORTH: Directions.SOUTH,
                Directions.SOUTH: Directions.NORTH,
                Directions.EAST: Directions.WEST,
                Directions.WEST: Directions.EAST,
            }

            non_reverse = []
            if self.last_dir is not None:
                for a in actions:
                    opposite = reverse_dir.get(self.last_dir)
                    if a != opposite:
                        non_reverse.append(a)

            if len(non_reverse) > 0:
                choice = random.choice(non_reverse)
            else:
                choice = random.choice(actions)

            self.last_dir = choice
            return choice

        # info of food, enemies, capsules
        food = []
        for f in self.get_food(game_state).as_list():
            food.append(to_int(f))

        capsules = []
        for c in self.get_capsules(game_state):
            capsules.append(to_int(c))

        enemies = []
        for i in self.get_opponents(game_state):
            enemies.append(game_state.get_agent_state(i))

        # separate visible enemies into ghosts (can kill us) and enemy pacman (invaders)
        ghosts = []
        enemy_pacmen = []
        for e in enemies:
            if e.get_position() is None:
                continue
            if e.is_pacman:
                enemy_pacmen.append(e)
            else:
                ghosts.append(e)

        carried = my_state.num_carrying
        retreat = False

        # near ghosts
        close_ghost = False
        for g in ghosts:
            gpos = to_int(g.get_position())
            dist = util.manhattan_distance(my_pos, gpos)
            if g.scared_timer == 0 and dist <= 4:
                close_ghost = True

        # defensive mode to help a bit with defense
        if self.is_home_side(my_pos) and len(enemy_pacmen) > 0:
            closest_enemy_pos = None
            closest_dist = 999999

            for e in enemy_pacmen:
                epos = to_int(e.get_position())
                d = util.manhattan_distance(my_pos, epos)
                if d < closest_dist:
                    closest_dist = d
                    closest_enemy_pos = epos

            if closest_enemy_pos is not None:
                if closest_dist <= 5:
                    path = self.astar(game_state, my_pos, [closest_enemy_pos])
                    if path:
                        nxt = path[0]
                        for a in game_state.get_legal_actions(self.index):
                            succ = game_state.generate_successor(self.index, a)
                            succ_pos = to_int(
                                succ.get_agent_state(self.index).get_position()
                            )
                            if succ_pos == nxt:
                                self.last_dir = a
                                return a

        # retreat triggers (i put 7 because with +7 score and the defense i have i will be able to win)
        if carried >= 7 or close_ghost:
            retreat = True
            self.was_retreating = True

        # scared ghosts
        scared_targets = []
        for g in ghosts:
            if g.get_position() is not None and g.scared_timer > 3:
                scared_targets.append(to_int(g.get_position()))

        # capsule rush: only when ghost is close and capsules exist
        capsule_rush = False
        if close_ghost and len(capsules) > 0:
            capsule_rush = True

        # select target by hierarchy
        target = None  # explicit default so we don't crash later

        if retreat:
            # avoid ghosts when pathfinding back to start
            avoid = []
            for g in ghosts:
                if g.get_position() is not None:
                    avoid.append(to_int(g.get_position()))

            path = self.astar(game_state, my_pos, [self.start], avoid_ghosts=avoid)
            if path:
                nxt = path[0]
                for a in game_state.get_legal_actions(self.index):
                    succ = game_state.generate_successor(self.index, a)
                    succ_pos = to_int(succ.get_agent_state(self.index).get_position())
                    if succ_pos == nxt:
                        self.last_dir = a
                        return a

            # if no path, fall back to using start as a target to avoid None
            target = [self.start]

        elif len(scared_targets) > 0:
            # go after scared ghosts if present and safe
            target = scared_targets

        elif capsule_rush:
            # try to eat a capsule if ghost is close (can flip the situation)
            target = capsules

        else:
            # prefer top and bottom food, as most probably enemy will be around mid defending
            top = []
            mid = []
            bot = []

            for f in food:
                fx, fy = f
                if fy > self.mid_y + 2:
                    top.append(f)
                elif fy < self.mid_y - 2:
                    bot.append(f)
                else:
                    mid.append(f)

            # find closest food
            def best(band):
                if len(band) == 0:
                    return (None, 999)
                best_food = None
                best_dist = 999
                for p in band:
                    d = util.manhattan_distance(my_pos, p)
                    if d < best_dist:
                        best_dist = d
                        best_food = p
                return (best_food, best_dist)

            t_food, t_d = best(top)
            b_food, b_d = best(bot)
            m_food, m_d = best(mid)

            # simple heuristics: bonus for edges and for groups  of food
            TOP_BONUS = 5
            BOT_BONUS = 5
            MID_BONUS = 0
            CLUSTER = 1.5

            def score(dist, bonus, size):
                # higher score == more attractive target
                return bonus - dist + size * CLUSTER

            food_s = []

            if t_food is not None:
                s = score(t_d, TOP_BONUS, len(top))
                food_s.append((t_food, s))

            if b_food is not None:
                s = score(b_d, BOT_BONUS, len(bot))
                food_s.append((b_food, s))

            if m_food is not None:
                s = score(m_d, MID_BONUS, len(mid))
                food_s.append((m_food, s))

            if len(food_s) > 0:
                # pick best scored food region (no lambda, no max with key)
                best_score = -999999
                best_food = None
                for fpos, s in food_s:
                    if s > best_score:
                        best_score = s
                        best_food = fpos
                target = [best_food]
            else:
                target = [self.start]

        # safe fallback
        if not target:
            target = [self.start]

        # A* for path
        path = self.astar(game_state, my_pos, target)
        if path:
            nxt = path[0]
            for a in game_state.get_legal_actions(self.index):
                succ = game_state.generate_successor(self.index, a)
                succ_pos = to_int(succ.get_agent_state(self.index).get_position())
                if succ_pos == nxt:
                    self.last_dir = a
                    return a

        # worst case random legal action
        a = random.choice(game_state.get_legal_actions(self.index))
        self.last_dir = a
        return a

    def get_features(self, s, a):
        return util.Counter()

    def get_weights(self, s, a):
        return util.Counter()


class DefensiveReflexAgent(ReflexCaptureAgent):

    # strategy is to camp around mid patrolling the border

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # decide patrol column based on team color
        # red defends left = mid_x - 1
        # blue defends right = mid_x
        if self.red:
            self.patrol_x = self.mid_x - 1
        else:
            self.patrol_x = self.mid_x

        # all vertical cells in that column that are not walls
        patrol_points = []
        for y in range(1, self.height - 1):
            if not self.walls[self.patrol_x][y]:
                patrol_points.append((self.patrol_x, y))

        # if column is fully blocked, just stay near spawn
        if len(patrol_points) == 0:
            self.patrol_points = [self.start]
        else:
            self.patrol_points = patrol_points

        # start patrolling in the middle of the chosen vertical segment
        self.patrol_index = len(self.patrol_points) // 2

        # patrol direction: +1 means going "down", -1 means going "up"
        self.patrol_dir = 1

    # basic reverse direction penalty, used when A* fallback
    def turn_penalty(self, prev, new):
        rev = {
            Directions.NORTH: Directions.SOUTH,
            Directions.SOUTH: Directions.NORTH,
            Directions.EAST: Directions.WEST,
            Directions.WEST: Directions.EAST,
        }
        if prev is None:
            return 0
        opposite = rev.get(prev)
        if new == opposite:
            return 5
        return 0

    def choose_action(self, game_state):
        my_pos = to_int(game_state.get_agent_state(self.index).get_position())

        # collect all legal moves, but try to stay as a GHOST at all costs
        legal = game_state.get_legal_actions(self.index)
        safe = []
        for a in legal:
            succ = game_state.generate_successor(self.index, a)
            succ_state = succ.get_agent_state(self.index)
            if not succ_state.is_pacman:
                safe.append(a)

        if len(safe) == 0:
            # if every move makes us Pacman, fallback to any legal move
            safe = legal

        # detect invaders
        enemies = []
        for i in self.get_opponents(game_state):
            enemies.append(game_state.get_agent_state(i))

        invaders = []
        for e in enemies:
            if e.is_pacman and e.get_position() is not None:
                invaders.append(e)

        #  PRIORITY 1: chase invader immediately
        if len(invaders) > 0:
            # pick the closest invader
            closest_invader_pos = None
            closest_dist = 999999

            for e in invaders:
                pos = to_int(e.get_position())
                d = util.manhattan_distance(my_pos, pos)
                if d < closest_dist:
                    closest_dist = d
                    closest_invader_pos = pos

            if closest_invader_pos is not None:
                path = self.astar(game_state, my_pos, [closest_invader_pos])
                if path:
                    nxt = path[0]
                    for a in safe:
                        succ = game_state.generate_successor(self.index, a)
                        succ_pos = to_int(
                            succ.get_agent_state(self.index).get_position()
                        )
                        if succ_pos == nxt:
                            self.last_dir = a
                            return a

            # couldn't find path, so move randomly
            a = random.choice(safe)
            self.last_dir = a
            return a

        #  PRIORITY 2: patrol up and down
        if len(self.patrol_points) > 1:
            self.patrol_index += self.patrol_dir

            # if we reach the bottom end of the patrol
            if self.patrol_index >= len(self.patrol_points):
                self.patrol_index = len(self.patrol_points) - 2
                self.patrol_dir = -1  # flip direction

            # if reach the top end of the patrol
            elif self.patrol_index < 0:
                self.patrol_index = 1
                self.patrol_dir = 1

        target = [self.patrol_points[self.patrol_index]]
        path = self.astar(game_state, my_pos, target)

        if path:
            nxt = path[0]
            # find the action that moves into that next tile
            for a in safe:
                succ = game_state.generate_successor(self.index, a)
                succ_pos = to_int(succ.get_agent_state(self.index).get_position())
                if succ_pos == nxt:
                    self.last_dir = a
                    return a

        # fallback if A* fails
        best = None
        best_score = 999999

        for a in safe:
            succ = game_state.generate_successor(self.index, a)
            spos = to_int(succ.get_agent_state(self.index).get_position())

            # how far horizontally are we from the patrol column
            dist_mid = abs(spos[0] - self.patrol_x)

            # add reverse-turn penalty (makes patrol feel smoother)
            turn = self.turn_penalty(self.last_dir, a)
            score = dist_mid + turn

            if score < best_score:
                best_score = score
                best = a

        self.last_dir = best
        return best

    # simple defaults — defensive agent uses no feature/weight learning
    def get_features(self, s, a):
        return util.Counter()

    def get_weights(self, s, a):
        return util.Counter()
