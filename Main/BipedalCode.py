# CODE ENVIRONNEMENT recompense
reward = 0

shaping = (
    130 * pos[0] / SCALE
)  # moving forward is a way to receive reward (normalized to get 300 on completion)
shaping -= 5.0 * abs(
    state[0]
)  # keep head straight, other than that and falling, any behavior is unpunished

reward = 0
if self.prev_shaping is not None:
    reward = shaping - self.prev_shaping
self.prev_shaping = shaping

for a in action:
    reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
    # normalized to about -50.0 using heuristic, more optimal agent should spend less

terminated = False
if self.game_over or pos[0] < 0:
    reward = -100
    terminated = True
if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
    terminated = True



# OBSERVATIONS
pos = self.hull.position
vel = self.hull.linearVelocity

# Lidar
for i in range(10):
    self.lidar[i].fraction = 1.0
    self.lidar[i].p1 = pos
    self.lidar[i].p2 = (
        pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
        pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
    )
    # calcule deux points puit on fait un raycast
    self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

state = [
    self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
    2.0 * self.hull.angularVelocity / FPS,
    0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
    0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
    self.joints[0].angle,
    # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
    self.joints[0].speed / SPEED_HIP,
    self.joints[1].angle + 1.0,
    self.joints[1].speed / SPEED_KNEE,
    1.0 if self.legs[1].ground_contact else 0.0,
    self.joints[2].angle,
    self.joints[2].speed / SPEED_HIP,
    self.joints[3].angle + 1.0,
    self.joints[3].speed / SPEED_KNEE,
    1.0 if self.legs[3].ground_contact else 0.0,
]