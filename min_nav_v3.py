"""AI2-THOR metadata-only navigation baseline — Graph Search Oracle (Plan A)."""

from __future__ import annotations

import argparse
import collections
import math
import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ai2thor.controller import Controller

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
class NavState(str, Enum):
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    ALIGN = "ALIGN"
    STOP = "STOP"
    FAIL = "FAIL"

@dataclass
class NavConfig:
    scene: str = "FloorPlan1"
    target_object_type: str = "Apple"

    # ── success criteria ─────────────────────────────────────
    distance_threshold: float = 1.5
    center_min: float = 0.30
    center_max: float = 0.70

    # ── spawn filter ─────────────────────────────────────────
    spawn_min_distance: float = 2.5
    spawn_max_trials: int = 200

    # ── controller ───────────────────────────────────────────
    grid_size: float = 0.25
    rotate_step_degrees: int = 30
    fine_rotate_degrees: float = 15.0  # ALIGN uses smaller rotations
    visibility_distance: float = 1.5
    width: int = 600
    height: int = 600
    headless: bool = True
    render_instance_segmentation: bool = True

    # ── episode budget ───────────────────────────────────────
    max_steps: int = 300

    # ── reproducibility ──────────────────────────────────────
    seed: int = 0

    # ── output ───────────────────────────────────────────────
    save_gif: bool = True
    gif_fps: int = 5
    gif_dir: str = "outputs/graph_nav_gifs"

# ────────────────────────────────────────────────────────────────
# Navigator (Graph-Based BFS Oracle)
# ────────────────────────────────────────────────────────────────
class MetadataNavigator:
    def __init__(
        self,
        controller: Controller,
        config: NavConfig,
        gif_path: Optional[Path],
        rng: random.Random,
    ) -> None:
        self.controller = controller
        self.config = config
        self.gif_path = gif_path
        self.rng = rng

        self.frames: List = []
        self.state: NavState = NavState.SEARCH

        # state counters
        self.state_steps = {s: 0 for s in NavState}

        # Graph Search & Pathfinding Variables
        self._reachable_nodes: Set[Tuple[int, int]] = set()
        self._graph: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._build_nav_graph()

        self.visited_poses: Set[Tuple[int, int]] = set()
        self.current_path: List[Tuple[int, int]] = []
        self.target_goal_node: Optional[Tuple[int, int]] = None
        
        self.last_intended_node: Optional[Tuple[int, int]] = None
        self.align_spins: int = 0
        self._bump_cooldown: int = 0
        self._pending_yaw: Optional[float] = None  # target yaw for teleport rotation

    # ─── Graph Construction ─────────────────────────────────
    def _to_grid(self, pos: Dict) -> Tuple[int, int]:
        """Convert a 3D float position into a stable 2D integer grid tuple."""
        return (int(round(pos["x"] / self.config.grid_size)),
                int(round(pos["z"] / self.config.grid_size)))

    def _build_nav_graph(self):
        """Builds a 4-connected undirected graph using reachable grid positions."""
        ev = self.controller.step(action="GetReachablePositions")
        reachable = ev.metadata.get("actionReturn") or []
        
        self._reachable_nodes = set(self._to_grid(p) for p in reachable)
        self._graph = {node: [] for node in self._reachable_nodes}
        
        for ix, iz in self._reachable_nodes:
            # 4-way connections
            for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxt = (ix + dx, iz + dz)
                if nxt in self._reachable_nodes:
                    self._graph[(ix, iz)].append(nxt)

    def _bfs(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Standard Breadth-First Search to find the shortest path in the grid map."""
        if start not in self._graph or goal not in self._graph:
            return []
            
        queue = collections.deque([[start]])
        visited = set([start])
        
        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr == goal:
                return path
            for nxt in self._graph.get(curr, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return []

    # ─── frame / gif ────────────────────────────────────────
    def _record_frame(self, event) -> None:
        if self.config.save_gif and event is not None and hasattr(event, "frame") and event.frame is not None:
            self.frames.append(event.frame)

    def _save_gif(self) -> None:
        if not self.config.save_gif or self.gif_path is None or not self.frames:
            return
        self.gif_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio
            imageio.mimsave(self.gif_path, self.frames, fps=self.config.gif_fps)
        except (ModuleNotFoundError, ImportError):
            from PIL import Image
            pil = [Image.fromarray(f) for f in self.frames]
            pil[0].save(
                self.gif_path, save_all=True, append_images=pil[1:],
                duration=int(1000 / self.config.gif_fps), loop=0,
            )
        print(f"[Info] GIF saved: {self.gif_path.resolve()}")

    # ─── perception ─────────────────────────────────────────
    def _nearest_target(self, event) -> Optional[Dict]:
        targets = [
            o for o in event.metadata.get("objects", [])
            if o.get("objectType") == self.config.target_object_type
        ]
        if not targets:
            return None
        return min(targets, key=lambda o: o.get("distance", float("inf")))

    def _screen_x_ratio(self, event, obj: Dict) -> Optional[float]:
        object_id = obj.get("objectId")
        detections = getattr(event, "instance_detections2D", None)
        if isinstance(detections, dict) and object_id in detections:
            bbox = detections[object_id]
            x1, _, x2, _ = bbox
            cx = (x1 + x2) / 2.0
            return max(0.0, min(1.0, cx / float(self.config.width)))

        return self._project_to_screen(event, obj)

    def _project_to_screen(self, event, obj: Dict) -> Optional[float]:
        pos = obj.get("position")
        if not isinstance(pos, dict):
            return None
        px, pz = float(pos["x"]), float(pos["z"])

        agent_meta = event.metadata["agent"]
        cam_pos = event.metadata.get("cameraPosition", agent_meta["position"])
        yaw = float(agent_meta["rotation"]["y"])

        dx = px - float(cam_pos["x"])
        dz = pz - float(cam_pos["z"])

        yaw_rad = math.radians(yaw)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        cam_x = dx * cos_y - dz * sin_y
        cam_z = dx * sin_y + dz * cos_y

        if cam_z <= 1e-6:
            # Object behind camera plane — return edge hint based on lateral offset
            return 0.0 if cam_x < 0 else 1.0

        fov = float(event.metadata.get("fov", 90.0))
        half_w = cam_z * math.tan(math.radians(fov) / 2.0)
        ndc = cam_x / half_w
        return max(0.0, min(1.0, (ndc + 1.0) / 2.0))

    def _target_obs(self, event) -> Tuple[bool, Optional[float], Optional[float]]:
        obj = self._nearest_target(event)
        if obj is None or not obj.get("visible"):
            return False, None, None
        dist = float(obj.get("distance"))
        sx = self._screen_x_ratio(event, obj)
        return True, dist, sx

    def _is_centered(self, sx: Optional[float]) -> bool:
        return sx is not None and self.config.center_min <= sx <= self.config.center_max

    # ─── execution wrappers ─────────────────────────────────
    def _fine_rotate(self, delta_deg: float):
        """Rotate using Teleport to allow arbitrary fine angles."""
        agent = self.controller.last_event.metadata["agent"]
        new_yaw = (float(agent["rotation"]["y"]) + delta_deg) % 360.0
        return self.controller.step(
            action="Teleport",
            position=agent["position"],
            rotation={"x": 0, "y": new_yaw, "z": 0},
            horizon=agent.get("cameraHorizon", 0),
            standing=True,
        )

    def _get_action_to_node(self, curr_node: Tuple[int, int], yaw: float, next_node: Tuple[int, int]) -> Tuple[str, Optional[float]]:
        """Returns (action, target_yaw) to traverse the graph edge.
        action is 'MoveAhead' or 'Rotate'; target_yaw is the desired facing angle."""
        dx = next_node[0] - curr_node[0]
        dz = next_node[1] - curr_node[1]

        # Calculate target yaw based on AI2-THOR's global axis
        if dx == 1 and dz == 0: target_yaw = 90.0
        elif dx == -1 and dz == 0: target_yaw = 270.0
        elif dx == 0 and dz == 1: target_yaw = 0.0
        elif dx == 0 and dz == -1: target_yaw = 180.0
        else:
            return "MoveAhead", None  # Fallback

        diff = (target_yaw - round(yaw)) % 360.0

        # Give 5 degrees tolerance due to floating point rotation
        if diff < 5 or diff > 355:
            return "MoveAhead", None
        else:
            return "Rotate", target_yaw

    # ─── decision ───────────────────────────────────────────
    def _decide(self, event, visible: bool, dist: Optional[float], sx: Optional[float]) -> str:
        pos = event.metadata["agent"]["position"]
        yaw = float(event.metadata["agent"]["rotation"]["y"])
        curr_node = self._to_grid(pos)
        centered = self._is_centered(sx)

        # ── Global STOP check ──
        if visible and dist is not None and dist < self.config.distance_threshold and centered:
            self.state = NavState.STOP
            return "Stop"

        # ── SEARCH ──
        if self.state == NavState.SEARCH:
            self.state_steps[NavState.SEARCH] += 1

            # Decrement bump cooldown
            if self._bump_cooldown > 0:
                self._bump_cooldown -= 1

            # Target Found! Compute a graph path directly to the target
            if visible and self._bump_cooldown <= 0:
                self.state = NavState.APPROACH
                target_obj = self._nearest_target(event)
                t_pos = target_obj["position"]
                
                # Find the reachable grid node closest to the target
                best_node = None
                min_d = float('inf')
                for node in self._reachable_nodes:
                    nx = node[0] * self.config.grid_size
                    nz = node[1] * self.config.grid_size
                    d = math.hypot(nx - t_pos["x"], nz - t_pos["z"])
                    if d < min_d:
                        min_d = d
                        best_node = node

                self.target_goal_node = best_node
                self.current_path = self._bfs(curr_node, self.target_goal_node)
                if self.current_path:
                    self.current_path.pop(0) # Remove starting node
                return self._decide(event, visible, dist, sx)

            # Still searching - we navigate to the nearest unvisited node
            if not self.current_path:
                unvisited = self._reachable_nodes - self.visited_poses
                if not unvisited:
                    if self._bump_cooldown > 0:
                        # Reset exploration to try different approach angles
                        self.visited_poses.clear()
                        unvisited = self._reachable_nodes
                    else:
                        return "FAIL_NO_UNVISITED"
                
                # Sort by straight-line distance to pick the nearest unvisited region
                best_unv = min(unvisited, key=lambda n: math.hypot(n[0]-curr_node[0], n[1]-curr_node[1]))
                self.current_path = self._bfs(curr_node, best_unv)
                
                if self.current_path:
                    self.current_path.pop(0)
                    if not self.current_path:
                        # Path was just the current node (already there)
                        self.visited_poses.add(best_unv)
                        return self._decide(event, visible, dist, sx)
                else:
                    self.visited_poses.add(best_unv) # Unreachable, skip
                    return self._decide(event, visible, dist, sx)

            # Execute path traversal
            next_node = self.current_path[0]
            action, target_yaw = self._get_action_to_node(curr_node, yaw, next_node)
            if action == "MoveAhead":
                self.last_intended_node = next_node
            elif target_yaw is not None:
                self._pending_yaw = target_yaw
            return action

        # ── APPROACH ──
        # Pure visual servoing: ignore grid path, center the target and walk toward it.
        if self.state == NavState.APPROACH:
            self.state_steps[NavState.APPROACH] += 1

            if not visible:
                # Target lost — replan via grid or fall back to SEARCH
                if self.target_goal_node is not None:
                    self.current_path = self._bfs(curr_node, self.target_goal_node)
                    if self.current_path:
                        self.current_path.pop(0)
                if not self.current_path:
                    self.state = NavState.SEARCH
                    return self._decide(event, visible, dist, sx)
                # Follow grid path blindly for one step toward last known position
                next_node = self.current_path[0]
                action, target_yaw = self._get_action_to_node(curr_node, yaw, next_node)
                if action == "MoveAhead":
                    self.last_intended_node = next_node
                elif target_yaw is not None:
                    self._pending_yaw = target_yaw
                return action

            # Target is visible — visual servoing
            if dist is not None and dist < self.config.distance_threshold and centered:
                self.state = NavState.STOP
                return "Stop"

            if not centered:
                # Compute exact yaw to face target, teleport-rotate in one step
                target_obj = self._nearest_target(event)
                if target_obj:
                    t_pos = target_obj.get("position")
                    if isinstance(t_pos, dict):
                        px, pz = float(t_pos["x"]), float(t_pos["z"])
                        agent_pos = event.metadata["agent"]["position"]
                        ax, az = float(agent_pos["x"]), float(agent_pos["z"])
                        target_yaw = math.degrees(math.atan2(px - ax, pz - az)) % 360.0
                        self._pending_yaw = target_yaw
                        return "Rotate"
                # Fallback: fine rotate
                return "FineRight"

            # Visible and centered — walk forward
            self.current_path = []  # clear any stale grid path
            return "MoveAhead"

        # ── ALIGN (lightweight fallback, should rarely trigger now) ──
        if self.state == NavState.ALIGN:
            self.state_steps[NavState.ALIGN] += 1

            if not visible or not centered:
                if visible and sx is not None:
                    dx = sx - 0.5
                    return "FineRight" if dx > 0 else "FineLeft"
                self.align_spins += 1
                if self.align_spins > 24:
                    return "FAIL_ALIGN"
                return "FineRight" if visible else "FineRight"

        return "Stop"

    # ─── run ────────────────────────────────────────────────
    def run(self, run_idx: int) -> Dict:
        event = self.controller.last_event
        self._record_frame(event)

        steps = 0
        fail_reason = "STEP_BUDGET"

        while steps < self.config.max_steps:
            steps += 1
            visible, dist, sx = self._target_obs(event)
            action = self._decide(event, visible, dist, sx)

            d_str = f"{dist:.3f}" if dist is not None else "  N/A"
            sx_str = f"{sx:.3f}" if sx is not None else " N/A"
            print(
                f"Run {run_idx:02d} | Step {steps:03d} | "
                f"State: {self.state.value:8s} | Action: {action:18s} | "
                f"Vis: {visible!s:5s} | Dist: {d_str:>6s} | ScreenX: {sx_str}"
            )

            # Failure States
            if action == "Stop":
                self._save_gif()
                return {
                    "success": True, "steps": steps, "fail_reason": None,
                    "state_steps": self.state_steps.copy(),
                }
            elif action.startswith("FAIL"):
                fail_reason = action
                break

            # Execution logic
            if action == "FineLeft":
                event = self._fine_rotate(-self.config.fine_rotate_degrees)
            elif action == "FineRight":
                event = self._fine_rotate(self.config.fine_rotate_degrees)
            elif action == "Rotate" and self._pending_yaw is not None:
                # Teleport-rotate directly to target yaw (1 step instead of multiple 30° steps)
                agent = event.metadata["agent"]
                event = self.controller.step(
                    action="Teleport",
                    position=agent["position"],
                    rotation={"x": 0, "y": self._pending_yaw, "z": 0},
                    horizon=agent.get("cameraHorizon", 0),
                    standing=True,
                )
                self._pending_yaw = None
            else:
                event = self.controller.step(action=action)

                # Physics Collisions (Bump) Detection and Graph Self-Correction
                if action == "MoveAhead":
                    if event.metadata.get("lastActionSuccess", True):
                        if self.current_path:
                            self.visited_poses.add(self.current_path[0])
                            self.current_path.pop(0)
                    else:
                        print("[Warning] Agent Bumped! Dynamically severing graph edge and replanning...")
                        u = self._to_grid(event.metadata["agent"]["position"])
                        v = self.last_intended_node

                        # Remove the blocked path edge from graph to prevent infinite loops
                        if v and v in self._graph.get(u, []): self._graph[u].remove(v)
                        if v and u in self._graph.get(v, []): self._graph[v].remove(u)

                        # Clear path and switch to graph search to route around obstacle
                        self.current_path = []
                        if self.state == NavState.APPROACH:
                            self.state = NavState.SEARCH
                            self._bump_cooldown = 5  # explore for a few steps before re-approaching

            self._record_frame(event)

        self._save_gif()
        self.state = NavState.FAIL
        return {
            "success": False, "steps": steps, "fail_reason": fail_reason,
            "state_steps": self.state_steps.copy(),
        }

# ────────────────────────────────────────────────────────────────
# Controller / spawning
# ────────────────────────────────────────────────────────────────
def build_controller(config: NavConfig) -> Controller:
    kwargs = dict(
        scene=config.scene,
        gridSize=config.grid_size,
        rotateStepDegrees=config.rotate_step_degrees,
        visibilityDistance=config.visibility_distance,
        width=config.width,
        height=config.height,
        snapToGrid=False,
        renderInstanceSegmentation=config.render_instance_segmentation,
    )
    if config.headless:
        kwargs["platform"] = "CloudRendering"
        kwargs["server_start_timeout"] = 600.0
        kwargs["server_timeout"] = 300.0
        # headless=True disables server-side screenshot, but CloudRendering
        # doesn't need X11 even with headless=False (uses Vulkan/EGL).
        return Controller(headless=False, **kwargs)

    return Controller(headless=config.headless, **kwargs)

def _nearest_target_info(event, target_type: str) -> Tuple[bool, Optional[float]]:
    targets = [o for o in event.metadata.get("objects", []) if o.get("objectType") == target_type]
    if not targets:
        return False, None
    nearest = min(targets, key=lambda o: o.get("distance", float("inf")))
    return bool(nearest.get("visible", False)), float(nearest.get("distance", float("inf")))

def randomize_agent_start(
    controller: Controller, config: NavConfig, rng: random.Random
) -> bool:
    """Spawn so that target is NOT visible AND distance >= spawn_min_distance."""
    ev = controller.step(action="GetReachablePositions")
    positions = ev.metadata.get("actionReturn") or []
    if not positions:
        raise RuntimeError("GetReachablePositions returned empty list.")

    cands = positions[:]
    rng.shuffle(cands)
    trials = 0

    for pos in cands:
        yaws = [0, 90, 180, 270]
        rng.shuffle(yaws)
        for yaw in yaws:
            trials += 1
            controller.step(
                action="Teleport",
                position=pos,
                rotation={"x": 0, "y": yaw, "z": 0},
                horizon=0, standing=True,
            )
            visible, dist = _nearest_target_info(
                controller.last_event, config.target_object_type
            )
            if (not visible) and (dist is not None and dist >= config.spawn_min_distance):
                print(
                    f"Spawn OK | pos=({pos['x']:.2f},{pos['z']:.2f}) | "
                    f"yaw={yaw} | dist={dist:.2f}"
                )
                return True
            if trials >= config.spawn_max_trials:
                break
        if trials >= config.spawn_max_trials:
            break

    print("[Warning] spawn filter exhausted; episode marked INVALID.")
    return False

# ────────────────────────────────────────────────────────────────
# CLI / main
# ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI2-THOR Graph Search Navigator (Plan A)")
    p.add_argument("--no-headless", action="store_true", help="Use local rendering")
    p.add_argument("--gif-fps", type=int, default=5)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scene", type=str, default="FloorPlan1")
    p.add_argument("--target", type=str, default="Apple")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    config = NavConfig(
        scene=args.scene,
        target_object_type=args.target,
        headless=not args.no_headless,
        save_gif=not args.no_gif,
        gif_fps=args.gif_fps,
        seed=args.seed,
    )
    gif_root = Path(config.gif_dir)
    gif_root.mkdir(parents=True, exist_ok=True)


    controller = build_controller(config)
    results = []

    try:
        for run_idx in range(1, args.runs + 1):
            rng = random.Random(config.seed + run_idx)
            controller.reset(scene=config.scene)

            ok = randomize_agent_start(controller, config, rng)
            if not ok:
                results.append({"success": False, "steps": 0,
                                "fail_reason": "INVALID_SPAWN",
                                "state_steps": {s: 0 for s in NavState}})
                continue

            gif_path = gif_root / f"run_{run_idx:02d}.gif" if config.save_gif else None
            nav = MetadataNavigator(controller, config, gif_path, rng)
            results.append(nav.run(run_idx))
    finally:
        controller.stop()

    # ── summary ──
    n = len(results)
    succ = sum(1 for r in results if r["success"])
    avg_steps = sum(r["steps"] for r in results) / max(1, n)

    from collections import Counter
    fail_reasons = Counter(r["fail_reason"] for r in results if not r["success"])

    print("\n=== Summary ===")
    print(f"Success: {succ}/{n} = {succ / max(1, n):.2%}")
    print(f"Average steps: {avg_steps:.2f}")
    if fail_reasons:
        print("Failure breakdown:")
        for k, v in fail_reasons.most_common():
            print(f"  {k}: {v}")
    print(f"GIF dir: {gif_root.resolve()}")

if __name__ == "__main__":
    main()