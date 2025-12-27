import json
import math
import time

class Vec3:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def to_dict(self):
        return {"x": round(self.x, 3), "y": round(self.y, 3), "z": round(self.z, 3)}

class Wall:
    def __init__(self, name, color, axis, position, bounds, restitution=0.7, friction=0.3):
        self.name = name
        self.color = color
        self.axis = axis  # 'x', 'y', or 'z' - the axis this wall blocks
        self.position = position  # position on that axis
        self.bounds = bounds  # dict with other two axes and their ranges
        self.restitution = restitution  # bounce coefficient (force resistance)
        self.friction = friction  # tangential damping
        self.mesh = self._generate_mesh()

    def _generate_mesh(self):
        # Generate quad mesh (4 vertices, 2 triangles)
        axes_order = ['x', 'y', 'z']
        other_axes = [a for a in axes_order if a != self.axis]

        # Generate 4 corners
        corners = []
        for v0 in self.bounds[other_axes[0]]:
            for v1 in self.bounds[other_axes[1]]:
                pt = {'x': 0, 'y': 0, 'z': 0}
                pt[self.axis] = self.position
                pt[other_axes[0]] = v0
                pt[other_axes[1]] = v1
                corners.append({"x": pt['x'], "y": pt['y'], "z": pt['z']})

        # Normal points inward (toward room center)
        normal = {'x': 0, 'y': 0, 'z': 0}
        normal[self.axis] = -1 if self.position > 0 else 1

        return {
            "vertices": corners,
            "triangles": [[0, 1, 2], [2, 1, 3]],  # Two triangles forming quad
            "normal": normal
        }

    def to_state(self):
        return {
            "name": self.name,
            "type": "wall",
            "color": self.color,
            "axis": self.axis,
            "position": self.position,
            "restitution": self.restitution,
            "friction": self.friction,
            "mesh": self.mesh
        }

class PhysicsObject:
    def __init__(self, name, pos, vel=None, mass=1.0, color="white", shape="sphere", radius=0.5):
        self.name = name
        self.position = pos
        self.velocity = vel or Vec3()
        self.mass = mass
        self.color = color
        self.shape = shape
        self.radius = radius
        self.contacts = []
        self.grounded = False
        self.last_contact_force = Vec3()

    def check_wall_collision(self, wall):
        """Check and resolve collision with a wall, applying its force resistance"""
        pos_on_axis = getattr(self.position, wall.axis)
        vel_on_axis = getattr(self.velocity, wall.axis)

        # Determine collision threshold based on wall position
        if wall.axis == 'y' and wall.position == 0:  # Floor
            threshold = self.radius
            colliding = pos_on_axis <= threshold
        elif wall.position > 0:  # Positive walls (east, north, ceiling)
            threshold = wall.position - self.radius
            colliding = pos_on_axis >= threshold
        else:  # Negative walls (west, south)
            threshold = wall.position + self.radius
            colliding = pos_on_axis <= threshold

        if colliding:
            # Clamp position
            if wall.axis == 'y' and wall.position == 0:
                setattr(self.position, wall.axis, self.radius)
            elif wall.position > 0:
                setattr(self.position, wall.axis, threshold)
            else:
                setattr(self.position, wall.axis, threshold)

            # Apply restitution (bounce) and friction
            new_vel = -vel_on_axis * wall.restitution
            setattr(self.velocity, wall.axis, new_vel)

            # Apply friction to tangential velocity
            for other_axis in ['x', 'y', 'z']:
                if other_axis != wall.axis:
                    tangent_vel = getattr(self.velocity, other_axis)
                    setattr(self.velocity, other_axis, tangent_vel * (1 - wall.friction))

            # Record contact with force info
            impact_force = abs(vel_on_axis) * self.mass * (1 + wall.restitution)
            self.contacts.append({
                "wall": wall.name,
                "color": wall.color,
                "impact_force": round(impact_force, 3),
                "restitution_applied": wall.restitution
            })

            # Check if settled on floor
            if wall.axis == 'y' and wall.position == 0:
                if abs(self.velocity.y) < 0.3:
                    self.velocity.y = 0
                    self.grounded = True

            return True
        return False

    def update(self, dt, gravity, walls):
        self.contacts = []

        # Apply gravity if not grounded
        if not self.grounded:
            self.velocity.y += gravity * dt

        # Update position
        self.position = self.position + self.velocity * dt

        # Check all wall collisions
        self.grounded = False
        for wall in walls:
            self.check_wall_collision(wall)

    def to_state(self):
        return {
            "name": self.name,
            "shape": self.shape,
            "color": self.color,
            "radius": self.radius,
            "mass": self.mass,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "speed": round(self.velocity.magnitude(), 3),
            "grounded": self.grounded,
            "contacts": self.contacts
        }

class PhysicsWorld:
    def __init__(self):
        self.objects = []
        self.tick = 0
        self.time = 0.0
        self.gravity = -9.8

        # Create walls with meshes, colors, and force resistance
        self.walls = [
            Wall("floor", "dark_gray", "y", 0,
                 {'x': [-5, 5], 'z': [-5, 5]},
                 restitution=0.5, friction=0.4),
            Wall("ceiling", "white", "y", 10,
                 {'x': [-5, 5], 'z': [-5, 5]},
                 restitution=0.3, friction=0.1),
            Wall("wall_north", "blue", "z", 5,
                 {'x': [-5, 5], 'y': [0, 10]},
                 restitution=0.7, friction=0.2),
            Wall("wall_south", "red", "z", -5,
                 {'x': [-5, 5], 'y': [0, 10]},
                 restitution=0.7, friction=0.2),
            Wall("wall_east", "green", "x", 5,
                 {'y': [0, 10], 'z': [-5, 5]},
                 restitution=0.8, friction=0.15),  # Bouncy wall!
            Wall("wall_west", "yellow", "x", -5,
                 {'y': [0, 10], 'z': [-5, 5]},
                 restitution=0.6, friction=0.3),
        ]

    def add_object(self, obj):
        self.objects.append(obj)

    def step(self, dt=1/30):
        self.tick += 1
        self.time += dt
        for obj in self.objects:
            obj.update(dt, self.gravity, self.walls)

    def get_state(self):
        return {
            "tick": self.tick,
            "time": round(self.time, 3),
            "environment": {
                "bounds": {"x": [-5, 5], "y": [0, 10], "z": [-5, 5]},
                "gravity": self.gravity,
                "walls": [w.to_state() for w in self.walls]
            },
            "objects": [obj.to_state() for obj in self.objects]
        }

# Create world
world = PhysicsWorld()

# Add a bunny (represented as sphere for physics)
bunny = PhysicsObject(
    "bunny",
    Vec3(0, 4, 0),
    Vec3(3, 5, 2),  # Faster initial velocity to hit walls
    mass=1.0,
    color="pink",
    shape="bunny_mesh",
    radius=0.4
)
world.add_object(bunny)

# Add a ball
ball = PhysicsObject(
    "ball",
    Vec3(-3, 7, -2),
    Vec3(4, 2, 3),  # Angled to hit multiple walls
    mass=0.5,
    color="orange",
    shape="sphere",
    radius=0.3
)
world.add_object(ball)

# Run simulation and output state stream
print("=== PHYSICS DATA STREAM START ===")
print("=== ENVIRONMENT ===")
for wall in world.walls:
    ws = wall.to_state()
    print(f"  {ws['name']} ({ws['color']}): axis={ws['axis']}, pos={ws['position']}, "
          f"restitution={ws['restitution']}, friction={ws['friction']}")
    print(f"    mesh: {len(ws['mesh']['vertices'])} verts, {len(ws['mesh']['triangles'])} tris")
print()

print("=== SIMULATION ===\n")
for i in range(90):  # 3 seconds at 30Hz
    state = world.get_state()
    print(f"--- TICK {state['tick']} (t={state['time']}s) ---")
    for obj in state['objects']:
        print(f"  {obj['name']} ({obj['color']} {obj['shape']}, m={obj['mass']}):")
        print(f"    pos: ({obj['position']['x']}, {obj['position']['y']}, {obj['position']['z']})")
        print(f"    vel: ({obj['velocity']['x']}, {obj['velocity']['y']}, {obj['velocity']['z']}) | speed: {obj['speed']}")
        if obj['contacts']:
            for c in obj['contacts']:
                print(f"    CONTACT: {c['wall']} ({c['color']}) - force={c['impact_force']}, bounce={c['restitution_applied']}")
        if obj['grounded']:
            print(f"    [GROUNDED]")
    print()
    world.step()

print("=== PHYSICS DATA STREAM END ===")
