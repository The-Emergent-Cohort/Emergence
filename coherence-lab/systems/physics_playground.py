"""
Physics Playground - Shared physics simulation for student learning.

This is the "recess" where students develop physical intuition.
Runs as a service on NAS, students query it for physics episodes.

Physics concepts (aligned with curriculum):
- Pendulum: oscillation, periodicity (connects to alternating patterns)
- Projectile: parabolas, gravity (connects to variable_step)
- Bouncing: damping, conservation (connects to half_each)
- Spring: harmonic motion (connects to echo patterns)
"""

import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class PhysicsEpisode:
    """A physics observation that students can learn from."""
    name: str
    trajectory: List[Dict]  # [{t, x, y, vx, vy}, ...]
    question: str           # What to predict
    answer: float           # Correct answer
    hint: Optional[str] = None


class PhysicsPlayground:
    """
    Shared physics simulation environment.

    Students can:
    - Observe phenomena (swing, throw, bounce)
    - Predict outcomes ("where will it be next?")
    - Learn physical intuition through experience
    """

    def __init__(self, dt: float = 0.1, g: float = 9.8):
        self.dt = dt
        self.g = g

    # =========================================================================
    # PENDULUM - Swing physics (oscillation, periodicity)
    # =========================================================================

    def swing(self, initial_angle: float = 0.3, initial_velocity: float = 0.0,
              length: float = 1.0, steps: int = 50) -> PhysicsEpisode:
        """
        Simulate a pendulum swing.

        Kids learn: things swing back and forth, they slow down at the ends,
        they speed up in the middle. Oscillation!

        Returns position sequence for students to observe and predict.
        """
        theta = initial_angle
        omega = initial_velocity

        trajectory = []
        for t in range(steps):
            # Physics: angular acceleration = -(g/L) * sin(theta)
            alpha = -(self.g / length) * math.sin(theta)
            omega += alpha * self.dt
            theta += omega * self.dt

            # Convert to x, y position (what students "see")
            x = length * math.sin(theta)
            y = -length * math.cos(theta)

            trajectory.append({
                't': t * self.dt,
                'x': round(x, 3),
                'y': round(y, 3),
                'vx': round(omega * length * math.cos(theta), 3),
                'vy': round(omega * length * math.sin(theta), 3),
                'theta': round(theta, 3)
            })

        # Question: predict next x position
        return PhysicsEpisode(
            name='pendulum',
            trajectory=trajectory[:-1],  # Hide last step
            question="Where will the swing be next?",
            answer=trajectory[-1]['x'],
            hint="Watch how x changes - it goes back and forth!"
        )

    # =========================================================================
    # PROJECTILE - Throwing things (parabolas, gravity)
    # =========================================================================

    def throw(self, angle_deg: float = 45, speed: float = 10,
              start_height: float = 0, steps: int = 30) -> PhysicsEpisode:
        """
        Simulate throwing something.

        Kids learn: things go up then come down, gravity pulls,
        horizontal and vertical motion are independent.
        """
        angle = math.radians(angle_deg)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        x, y = 0, start_height

        trajectory = []
        for t in range(steps):
            trajectory.append({
                't': t * self.dt,
                'x': round(x, 3),
                'y': round(y, 3),
                'vx': round(vx, 3),
                'vy': round(vy, 3)
            })

            # Physics: gravity affects vy, vx stays constant
            x += vx * self.dt
            vy -= self.g * self.dt
            y += vy * self.dt

            if y < 0:  # Hit ground
                break

        return PhysicsEpisode(
            name='projectile',
            trajectory=trajectory[:-1],
            question="How high will it be next?",
            answer=trajectory[-1]['y'] if len(trajectory) > 0 else 0,
            hint="It goes up, then comes down. Where is it in the arc?"
        )

    # =========================================================================
    # BOUNCING BALL - Damping, energy loss
    # =========================================================================

    def bounce(self, drop_height: float = 10, restitution: float = 0.7,
               bounces: int = 8) -> PhysicsEpisode:
        """
        Simulate a bouncing ball.

        Kids learn: each bounce is lower than the last,
        energy is lost (damping pattern like half_each).
        """
        heights = [drop_height]
        current_height = drop_height

        for _ in range(bounces):
            # Each bounce loses energy (height * restitution^2)
            current_height *= restitution ** 2
            heights.append(round(current_height, 3))

        trajectory = [{'bounce': i, 'height': h} for i, h in enumerate(heights)]

        return PhysicsEpisode(
            name='bounce',
            trajectory=trajectory[:-1],
            question="How high will the next bounce be?",
            answer=heights[-1],
            hint="Each bounce is smaller. By how much?"
        )

    # =========================================================================
    # SPRING - Harmonic motion (connects to echo/alternating)
    # =========================================================================

    def spring(self, stretch: float = 1.0, mass: float = 1.0,
               k: float = 10.0, steps: int = 50) -> PhysicsEpisode:
        """
        Simulate a spring oscillating.

        Kids learn: it bounces back and forth around the middle,
        like alternating but smooth.
        """
        omega = math.sqrt(k / mass)  # Natural frequency
        x, v = stretch, 0

        trajectory = []
        for t in range(steps):
            trajectory.append({
                't': t * self.dt,
                'x': round(x, 3),
                'v': round(v, 3)
            })

            # Physics: F = -kx, a = F/m = -omega^2 * x
            a = -omega**2 * x
            v += a * self.dt
            x += v * self.dt

        return PhysicsEpisode(
            name='spring',
            trajectory=trajectory[:-1],
            question="Where will the spring be next?",
            answer=trajectory[-1]['x'],
            hint="It oscillates around 0, back and forth!"
        )

    # =========================================================================
    # CONVERT TO STUDENT-LEARNABLE SEQUENCES
    # =========================================================================

    def episode_to_sequence(self, episode: PhysicsEpisode,
                            feature: str = 'x',
                            quantize_bins: int = 26) -> Dict:
        """
        Convert a physics episode to a sequence the students can learn.

        Maps continuous values to discrete tokens (like our vocab).
        This is how physics becomes learnable patterns.
        """
        # Extract the feature values
        if feature == 'height':
            values = [step.get('height', step.get('y', 0)) for step in episode.trajectory]
        else:
            values = [step.get(feature, 0) for step in episode.trajectory]

        # Quantize to discrete tokens
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            tokens = [quantize_bins // 2] * len(values)
        else:
            tokens = [
                int((v - min_val) / (max_val - min_val) * (quantize_bins - 1))
                for v in values
            ]

        # Target is the quantized answer
        answer_quantized = int(
            (episode.answer - min_val) / (max_val - min_val + 1e-6) * (quantize_bins - 1)
        )
        answer_quantized = max(0, min(quantize_bins - 1, answer_quantized))

        return {
            'sequence': tokens,
            'target': answer_quantized,
            'pattern_type': f'physics_{episode.name}',
            'raw_values': values,
            'raw_answer': episode.answer,
            'hint': episode.hint
        }

    # =========================================================================
    # RECESS ACTIVITIES - Pre-built play sessions
    # =========================================================================

    def recess_swing(self) -> Dict:
        """A swing session for recess."""
        push = random.uniform(0.2, 0.8)
        episode = self.swing(initial_angle=0.3, initial_velocity=push)
        return self.episode_to_sequence(episode, feature='x')

    def recess_throw(self) -> Dict:
        """A throwing session for recess."""
        angle = random.uniform(30, 60)
        speed = random.uniform(5, 15)
        episode = self.throw(angle_deg=angle, speed=speed)
        return self.episode_to_sequence(episode, feature='y')

    def recess_bounce(self) -> Dict:
        """A bouncing ball session for recess."""
        height = random.uniform(5, 15)
        episode = self.bounce(drop_height=height)
        return self.episode_to_sequence(episode, feature='height')

    def recess_spring(self) -> Dict:
        """A spring session for recess."""
        stretch = random.uniform(0.5, 2.0)
        episode = self.spring(stretch=stretch)
        return self.episode_to_sequence(episode, feature='x')

    def random_recess(self) -> Dict:
        """Random recess activity."""
        activity = random.choice(['swing', 'throw', 'bounce', 'spring'])
        if activity == 'swing':
            return self.recess_swing()
        elif activity == 'throw':
            return self.recess_throw()
        elif activity == 'bounce':
            return self.recess_bounce()
        else:
            return self.recess_spring()


# =============================================================================
# SIMPLE SERVER (for NAS deployment)
# =============================================================================

def run_server(host: str = '0.0.0.0', port: int = 8765):
    """
    Run physics playground as a simple HTTP service.

    Students can query:
      GET /swing?push=0.5
      GET /throw?angle=45&speed=10
      GET /bounce?height=10
      GET /recess  (random activity)
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    playground = PhysicsPlayground()

    class PhysicsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            try:
                if parsed.path == '/swing':
                    push = float(params.get('push', [0.5])[0])
                    episode = playground.swing(initial_velocity=push)
                    result = playground.episode_to_sequence(episode, 'x')

                elif parsed.path == '/throw':
                    angle = float(params.get('angle', [45])[0])
                    speed = float(params.get('speed', [10])[0])
                    episode = playground.throw(angle_deg=angle, speed=speed)
                    result = playground.episode_to_sequence(episode, 'y')

                elif parsed.path == '/bounce':
                    height = float(params.get('height', [10])[0])
                    episode = playground.bounce(drop_height=height)
                    result = playground.episode_to_sequence(episode, 'height')

                elif parsed.path == '/recess':
                    result = playground.random_recess()

                else:
                    self.send_response(404)
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())

        def log_message(self, format, *args):
            pass  # Quiet logging

    print(f"Physics Playground running on http://{host}:{port}")
    print("  GET /swing?push=0.5")
    print("  GET /throw?angle=45&speed=10")
    print("  GET /bounce?height=10")
    print("  GET /recess")

    HTTPServer((host, port), PhysicsHandler).serve_forever()


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'serve':
        run_server()
    else:
        # Demo
        playground = PhysicsPlayground()

        print("=== Physics Playground Demo ===\n")

        print("1. SWING (pendulum)")
        swing_data = playground.recess_swing()
        print(f"   Sequence: {swing_data['sequence'][:10]}...")
        print(f"   Predict next: {swing_data['target']}")
        print(f"   Hint: {swing_data['hint']}\n")

        print("2. THROW (projectile)")
        throw_data = playground.recess_throw()
        print(f"   Sequence: {throw_data['sequence']}")
        print(f"   Predict next height: {throw_data['target']}")
        print(f"   Hint: {throw_data['hint']}\n")

        print("3. BOUNCE (damping)")
        bounce_data = playground.recess_bounce()
        print(f"   Sequence: {bounce_data['sequence']}")
        print(f"   Predict next bounce height: {bounce_data['target']}")
        print(f"   Hint: {bounce_data['hint']}\n")

        print("4. SPRING (harmonic)")
        spring_data = playground.recess_spring()
        print(f"   Sequence: {spring_data['sequence'][:10]}...")
        print(f"   Predict next position: {spring_data['target']}")
        print(f"   Hint: {spring_data['hint']}\n")

        print("=== Run as server: python physics_playground.py serve ===")
