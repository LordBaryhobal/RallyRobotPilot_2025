from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from io import TextIOWrapper
from math import acos, radians, sqrt
from pathlib import Path
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")
MAX_FACES_PER_MESH = 100


class ReIndexer(Generic[T]):
    def __init__(self, lst: list[T]):
        self.lst: list[T] = lst
        self.indices: list[int] = []

    def get(self, old_i: int) -> tuple[int, T]:
        if old_i not in self.indices:
            self.indices.append(old_i)
        new_i: int = self.indices.index(old_i)
        return new_i, self.lst[old_i]

    def get_new_list(self) -> list[T]:
        return [self.lst[i] for i in self.indices]


class Simplifier(Generic[T]):
    def __init__(
        self, lst: list[T], cmp: Callable[[T, T, float], bool], thresh: float
    ) -> None:
        self.lst: list[T] = lst
        self.cmp: Callable[[T, T, float], bool] = cmp
        self.thresh: float = thresh
        self.indices: list[int] = list(range(len(self.lst)))
        self.simplify()

    def simplify(self):
        for i in range(len(self.lst)):
            for j in range(i + 1, len(self.lst)):
                if self.cmp(self.lst[i], self.lst[j], self.thresh):
                    self.indices[j] = self.indices[i]

    def map_many(self, indices: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(self.map(i) for i in indices)

    def map(self, idx: int) -> int:
        return self.indices[idx]

    def print_stats(self, name: str):
        before: int = len(self.lst)
        after: int = sum(int(i == j) for i, j in enumerate(self.indices))
        if before != after:
            print(f"Simplified {name}: {before} -> {after}")


@dataclass
class Vec2:
    x: float = 0
    y: float = 0

    @staticmethod
    def abs_diff(a: Vec2, b: Vec2, thresh: float) -> bool:
        return abs(a.x - b.x) < thresh and abs(a.y - b.y) < thresh


@dataclass
class Vec3:
    x: float = 0
    y: float = 0
    z: float = 0

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        return sqrt(self.dot(self))

    @staticmethod
    def abs_diff(a: Vec3, b: Vec3, thresh: float) -> bool:
        return (
            abs(a.x - b.x) < thresh
            and abs(a.y - b.y) < thresh
            and abs(a.z - b.z) < thresh
        )

    @staticmethod
    def angle_diff(a: Vec3, b: Vec3, thresh: float) -> bool:
        dot: float = a.dot(b)
        a_len: float = a.length()
        b_len: float = b.length()
        cos: float = dot / a_len / b_len
        return acos(max(-1, min(1, cos))) < thresh


@dataclass
class Face:
    pts: tuple[int, ...]
    normals: tuple[int, ...]
    uvs: tuple[int, ...]

    def share_edge(self, other: Face) -> bool:
        return len(set(self.pts).intersection(set(other.pts))) >= 2


@dataclass
class Mesh:
    pts: list[Vec3] = field(default_factory=list)
    normals: list[Vec3] = field(default_factory=list)
    uvs: list[Vec2] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)

    def write(self, file: TextIOWrapper):
        for p in self.pts:
            file.write(f"v {p.x} {p.y} {p.z}\n")
        for n in self.normals:
            file.write(f"vn {n.x} {n.y} {n.z}\n")
        for uv in self.uvs:
            file.write(f"vt {uv.x} {uv.y}\n")
        for face in self.faces:
            parts: list[tuple[int, int, int]] = zip(face.pts, face.uvs, face.normals)  # type: ignore
            parts2: list[str] = ["/".join(map(lambda i: str(i + 1), p)) for p in parts]
            file.write(f"f {' '.join(parts2)}\n")

    def extract(self, pts: list[Vec3], normals: list[Vec3], uvs: list[Vec2]):
        pts_idx: ReIndexer[Vec3] = ReIndexer(pts)
        normals_idx: ReIndexer[Vec3] = ReIndexer(normals)
        uvs_idx: ReIndexer[Vec2] = ReIndexer(uvs)
        for face in self.faces:
            face.pts = tuple(pts_idx.get(i)[0] for i in face.pts)
            face.normals = tuple(normals_idx.get(i)[0] for i in face.normals)
            face.uvs = tuple(uvs_idx.get(i)[0] for i in face.uvs)
        self.pts = pts_idx.get_new_list()
        self.normals = normals_idx.get_new_list()
        self.uvs = uvs_idx.get_new_list()

    def merge(self):
        pts_simplifier: Simplifier[Vec3] = Simplifier(self.pts, Vec3.abs_diff, 1)
        normals_simplifier: Simplifier[Vec3] = Simplifier(
            self.normals, Vec3.angle_diff, radians(1)
        )
        uvs_simplifier: Simplifier[Vec2] = Simplifier(self.uvs, Vec2.abs_diff, 0.001)

        for face in self.faces:
            face.pts = pts_simplifier.map_many(face.pts)
            face.normals = normals_simplifier.map_many(face.normals)
            face.uvs = uvs_simplifier.map_many(face.uvs)

        pts_simplifier.print_stats("vertices")
        normals_simplifier.print_stats("normals")
        uvs_simplifier.print_stats("uvs")

    def split(self) -> list[Mesh]:
        n_faces: int = len(self.faces)
        adjacent: list[set[int]] = [set() for _ in range(n_faces)]
        for i in range(n_faces):
            for j in range(i + 1, n_faces):
                if self.faces[i].share_edge(self.faces[j]):
                    adjacent[i].add(j)
                    adjacent[j].add(i)

        unassigned: set[int] = set(range(n_faces))
        submeshes: list[Mesh] = []

        while unassigned:
            cluster: list[int] = self.grow_cluster(
                unassigned, adjacent, MAX_FACES_PER_MESH
            )
            unassigned -= set(cluster)
            submesh: Mesh = Mesh(faces=[self.faces[i] for i in cluster])
            submesh.extract(self.pts, self.normals, self.uvs)
            submeshes.append(submesh)
            print(f"Cluster {len(submeshes)}: {len(cluster)} faces")

        return submeshes

    @staticmethod
    def grow_cluster(
        unassigned: set[int], adjacent: list[set[int]], target_size: int
    ) -> list[int]:
        seed: int = next(iter(unassigned))
        cluster: list[int] = []
        queue: deque[int] = deque([seed])
        in_queue: set[int] = {seed}
        while queue and len(cluster) < target_size:
            face_idx: int = queue.popleft()
            if face_idx not in unassigned:
                continue
            cluster.append(face_idx)
            for neighbor in adjacent[face_idx]:
                if neighbor in unassigned and neighbor not in in_queue:
                    queue.append(neighbor)
                    in_queue.add(neighbor)
        return cluster


@dataclass
class Object:
    name: str = "Unnamed"
    mesh: Mesh = field(default_factory=Mesh)
    path: Path = Path()

    def save(self, dir: Path):
        self.path = dir / f"{self.name.replace('.', '_')}.obj"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            f.write(f"o {self.name}\n")
            self.mesh.write(f)

    def split_mesh(self) -> list[Object]:
        meshes: list[Mesh] = self.mesh.split()

        return [Object(f"{self.name}/{i:03d}", m) for i, m in enumerate(meshes)]


class MeshSplitter:
    def __init__(self, path: Path, out_dir: Path):
        self.path: Path = path
        self.out_dir: Path = out_dir
        self.objects: list[Object] = []

    def load(self):
        self.objects = []
        obj: Optional[Object] = None
        with open(self.path, "r") as f:
            pts: list[Vec3] = []
            normals: list[Vec3] = []
            uvs: list[Vec2] = []

            for line in f.readlines():
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts: list[str] = line.split(" ")
                if parts[0] == "o":
                    obj = Object(" ".join(parts[1:]))
                    self.objects.append(obj)
                    continue
                if obj is None:
                    obj = Object()
                    self.objects.append(obj)
                if parts[0] == "v":
                    pts.append(Vec3(*map(float, parts[1:])))
                elif parts[0] == "vn":
                    normals.append(Vec3(*map(float, parts[1:])))
                elif parts[0] == "vt":
                    uvs.append(Vec2(*map(float, parts[1:])))
                elif parts[0] == "f":
                    parts2: list[list[int]] = [
                        list(map(lambda s: int(s) - 1, p.split("/"))) for p in parts[1:]
                    ]
                    pts_i, uvs_i, normals_i = zip(*parts2)
                    face: Face = Face(pts_i, normals_i, uvs_i)
                    obj.mesh.faces.append(face)

            print(f"{len(pts)} pts, {len(normals)} normals, {len(uvs)} uvs")
            for obj in self.objects:
                print(obj.name)
                obj.mesh.extract(pts, normals, uvs)
                obj.mesh.merge()

            i: int = 0
            while i < len(self.objects):
                obj = self.objects[i]
                if len(obj.mesh.faces) > MAX_FACES_PER_MESH:
                    objs: list[Object] = obj.split_mesh()
                    self.objects.pop(i)
                    self.objects.extend(objs)
                else:
                    i += 1

    def split(self, update_metadata: bool = False):
        self.load()
        self.out_dir.mkdir(exist_ok=True)
        for obj in self.objects:
            obj.save(self.out_dir)

        if update_metadata:
            dir: Path = self.path.parent
            meta_path: Path = dir / "track_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                meta["obstacles"] = [
                    {
                        "model": str(obj.path.relative_to(dir)),
                        "texture": "generalTex.png",
                    }
                    for obj in self.objects
                ]
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)
            else:
                print(f"Metadata file not found (looked at {meta_path})")


if __name__ == "__main__":
    root: Path = Path(__file__).parent.parent
    splitter: MeshSplitter = MeshSplitter(
        root / "assets" / "SimpleTrack" / "Obstacles.obj",
        root / "assets" / "SimpleTrack" / "obstacles",
    )
    splitter.split(True)
