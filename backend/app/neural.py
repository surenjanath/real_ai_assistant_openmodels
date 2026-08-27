"""The neural substrate: a live cognitive graph of the running assistant.

Every subsystem in J.A.R.V.I.S. — speech-in, intent, memory, the reasoning
cortex, the tool motor pool, the vocal tract — is modelled as a **node** in a
directed graph, and every hand-off between them is an **edge**. As a command
flows through the system the corresponding nodes *fire* and the edges carry a
travelling *action potential*, which the interface renders as a real neural
network lighting up along genuine signal paths.

This is not decoration: nothing pulses unless the code path actually ran. The
graph is the architecture diagram, drawn by the program about itself.

Wire protocol
-------------
``neural.graph``  once per client (and after any dynamic node is grown)::

    {"type": "neural.graph",
     "nodes": [{"id","label","region","layer","kind"}...],
     "edges": [{"id","from","to","weight"}...]}

``neural`` at ~20 Hz while there is activity (coalesced, never per-event)::

    {"type": "neural", "ts": float,
     "spikes": [[node_index, intensity], ...],
     "flows":  [[edge_index, intensity], ...],
     "regions": {"cortex": 0.82, ...}}

Coalescing matters: a streaming model fires the cortex thousands of times a
second. Publishing each one would drown the telemetry socket, so intensities
are accumulated into a frame buffer and flushed on a fixed tick.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .logbus import LogBus

#: Frames per second for the coalesced spike stream.
FLUSH_HZ = 20.0
#: How fast an un-refreshed node decays back to rest, per second.
DECAY_PER_S = 2.6


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    region: str
    layer: int
    kind: str = "core"  # core | dynamic | tool


@dataclass(frozen=True)
class Edge:
    id: str
    src: str
    dst: str
    weight: float = 1.0


# ---------------------------------------------------------------------------
# The standing architecture. Layers read left-to-right as signal flow:
#   0 sensory -> 1 intake -> 2 memory -> 3 cortex -> 4 effectors -> 5 motor
# ---------------------------------------------------------------------------

_NODES: list[Node] = [
    # -- afferent / sensory ------------------------------------------------
    Node("sense.text", "TEXT IN", "sensory", 0),
    Node("sense.voice", "SPEECH IN", "sensory", 0),
    Node("sense.host", "HOST SENSE", "sensory", 0),
    # -- intake / classification -------------------------------------------
    Node("intake.intent", "INTENT", "intake", 1),
    Node("intake.control", "CONTROL", "intake", 1),
    Node("intake.route", "ROUTER", "intake", 1),
    Node("intake.reflex", "REFLEX", "intake", 1),
    # -- hippocampus / memory ----------------------------------------------
    Node("mem.working", "WORKING", "memory", 2),
    Node("mem.recall", "RECALL", "memory", 2),
    Node("mem.write", "CONSOLIDATE", "memory", 2),
    # -- cortex / reasoning -------------------------------------------------
    Node("cortex.jarvis", "PERSONA", "cortex", 3),
    Node("cortex.analyst", "ANALYST", "cortex", 3),
    Node("cortex.engineer", "ENGINEER", "cortex", 3),
    Node("cortex.synth", "SYNTH", "cortex", 3),
    Node("cortex.think", "DELIBERATE", "cortex", 3),
    # -- effector pool (tools grow here at runtime) -------------------------
    Node("motor.tools", "TOOL BUS", "effector", 4),
    # -- efferent / motor ---------------------------------------------------
    Node("out.answer", "COMPOSE", "motor", 5),
    Node("out.tts", "VOCALISE", "motor", 5),
    Node("out.audio", "AUDIO OUT", "motor", 5),
]

_EDGES: list[Edge] = [
    Edge("e01", "sense.text", "intake.intent"),
    Edge("e02", "sense.voice", "intake.intent"),
    Edge("e03", "intake.intent", "intake.control", 0.7),
    Edge("e04", "intake.intent", "intake.route"),
    Edge("e05", "intake.route", "mem.working"),
    Edge("e06", "intake.route", "mem.recall", 0.8),
    Edge("e07", "mem.working", "cortex.jarvis"),
    Edge("e08", "mem.recall", "cortex.jarvis", 0.8),
    Edge("e09", "intake.route", "cortex.analyst", 0.6),
    Edge("e10", "cortex.analyst", "cortex.engineer", 0.6),
    Edge("e11", "cortex.engineer", "cortex.synth", 0.6),
    Edge("e12", "cortex.jarvis", "cortex.think", 0.5),
    Edge("e13", "cortex.think", "cortex.jarvis", 0.5),
    Edge("e14", "cortex.jarvis", "motor.tools", 0.7),
    Edge("e15", "cortex.engineer", "motor.tools", 0.5),
    Edge("e16", "motor.tools", "cortex.jarvis", 0.7),
    Edge("e17", "cortex.jarvis", "out.answer"),
    Edge("e18", "cortex.synth", "out.answer"),
    Edge("e19", "intake.control", "out.answer", 0.7),
    Edge("e20", "out.answer", "mem.write", 0.6),
    Edge("e21", "out.answer", "out.tts"),
    Edge("e22", "out.tts", "out.audio"),
    Edge("e23", "sense.host", "mem.working", 0.25),
    Edge("e24", "sense.host", "out.answer", 0.15),
    # The reflex arc: a spinal shortcut that grounds the answer before
    # the cortex has finished deciding what to do about it.
    Edge("e25", "intake.intent", "intake.reflex", 0.6),
    Edge("e26", "intake.reflex", "motor.tools", 0.6),
    Edge("e27", "intake.reflex", "cortex.jarvis", 0.8),
]

REGIONS = ("sensory", "intake", "memory", "cortex", "effector", "motor")


class NeuralBus:
    """Accumulates activation and flushes coalesced frames onto the LogBus."""

    def __init__(self, bus: LogBus) -> None:
        self.bus = bus
        self.nodes: list[Node] = list(_NODES)
        self.edges: list[Edge] = list(_EDGES)
        self._index: dict[str, int] = {n.id: i for i, n in enumerate(self.nodes)}
        self._edge_index: dict[str, int] = {e.id: i for i, e in enumerate(self.edges)}
        #: pending intensity per node/edge since the last flush
        self._spikes: dict[int, float] = {}
        self._flows: dict[int, float] = {}
        #: decayed activation level per node, 0..1 — what the HUD meters show
        self.activation: list[float] = [0.0] * len(self.nodes)
        self._task: asyncio.Task | None = None
        self._last_flush = time.monotonic()
        self._dirty_graph = False
        #: total spikes since boot, a cheap "neurons fired" odometer
        self.fired = 0

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="neural")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    # -- graph --------------------------------------------------------------

    def graph_frame(self) -> dict:
        return {
            "type": "neural.graph",
            "nodes": [
                {"id": n.id, "label": n.label, "region": n.region,
                 "layer": n.layer, "kind": n.kind}
                for n in self.nodes
            ],
            "edges": [
                {"id": e.id, "from": e.src, "to": e.dst, "weight": e.weight}
                for e in self.edges
            ],
        }

    def ensure_tool_node(self, tool: str) -> str:
        """Grow a node for a tool the first time it is actually invoked.

        The graph is therefore a record of capability that has really been
        exercised, not a catalogue of what might exist.
        """
        node_id = f"tool.{tool}"
        if node_id in self._index:
            return node_id
        self.nodes.append(Node(node_id, tool.replace("_", " ").upper(), "effector", 4, kind="tool"))
        self._index[node_id] = len(self.nodes) - 1
        self.activation.append(0.0)
        for src, dst, eid in (
            ("motor.tools", node_id, f"t-in-{tool}"),
            (node_id, "cortex.jarvis", f"t-out-{tool}"),
        ):
            self.edges.append(Edge(eid, src, dst, 0.6))
            self._edge_index[eid] = len(self.edges) - 1
        self._dirty_graph = True
        return node_id

    # -- firing -------------------------------------------------------------

    def fire(self, node_id: str, intensity: float = 1.0) -> None:
        """Excite one node. Safe to call from any thread, at any rate."""
        idx = self._index.get(node_id)
        if idx is None:
            return
        self.fired += 1
        prev = self._spikes.get(idx, 0.0)
        self._spikes[idx] = min(1.0, prev + max(0.0, intensity))

    def flow(self, edge_id: str, intensity: float = 1.0) -> None:
        idx = self._edge_index.get(edge_id)
        if idx is not None:
            self._flows[idx] = min(1.0, self._flows.get(idx, 0.0) + max(0.0, intensity))

    def signal(self, src: str, dst: str, intensity: float = 1.0) -> None:
        """Fire both endpoints and light the edge between them, if one exists."""
        self.fire(src, intensity)
        self.fire(dst, intensity)
        for edge in self.edges:
            if edge.src == src and edge.dst == dst:
                self.flow(edge.id, intensity)
                return

    def path(self, *node_ids: str, intensity: float = 1.0) -> None:
        """Propagate along a chain of nodes: path('a','b','c')."""
        for src, dst in zip(node_ids, node_ids[1:]):
            self.signal(src, dst, intensity)

    # -- flush --------------------------------------------------------------

    async def _loop(self) -> None:
        period = 1.0 / FLUSH_HZ
        while True:
            await asyncio.sleep(period)
            try:
                self._flush()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — the substrate must never die
                pass

    def _flush(self) -> None:
        now = time.monotonic()
        dt = max(1e-3, now - self._last_flush)
        self._last_flush = now

        spikes, self._spikes = self._spikes, {}
        flows, self._flows = self._flows, {}

        decay = max(0.0, 1.0 - DECAY_PER_S * dt)
        active = False
        for i in range(len(self.activation)):
            level = self.activation[i] * decay
            hit = spikes.get(i)
            if hit:
                level = min(1.0, level + hit)
            self.activation[i] = level if level > 0.004 else 0.0
            if self.activation[i] > 0.0:
                active = True

        if self._dirty_graph:
            self.bus.push_frame(self.graph_frame())
            self._dirty_graph = False

        if not spikes and not flows and not active:
            return

        regions: dict[str, float] = {r: 0.0 for r in REGIONS}
        for i, node in enumerate(self.nodes):
            regions[node.region] = max(regions.get(node.region, 0.0), self.activation[i])

        self.bus.push_frame({
            "type": "neural",
            "ts": time.time(),
            "spikes": [[i, round(v, 3)] for i, v in spikes.items()],
            "flows": [[i, round(v, 3)] for i, v in flows.items()],
            "levels": [round(v, 3) for v in self.activation],
            "regions": {k: round(v, 3) for k, v in regions.items()},
            "fired": self.fired,
        })


@dataclass
class NullNeural:
    """No-op stand-in so subsystems can hold an optional reference."""

    nodes: list = field(default_factory=list)
    fired: int = 0

    def fire(self, *_a, **_k) -> None: ...
    def flow(self, *_a, **_k) -> None: ...
    def signal(self, *_a, **_k) -> None: ...
    def path(self, *_a, **_k) -> None: ...
    def ensure_tool_node(self, tool: str) -> str: return f"tool.{tool}"
