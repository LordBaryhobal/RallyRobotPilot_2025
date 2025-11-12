from __future__ import annotations

from dataclasses import dataclass, field
from io import TextIOWrapper
import json
from pathlib import Path
from typing import Generic, Optional, TypeVar

T = TypeVar("T")

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


@dataclass
class Vec2:
    x: float = 0
    y: float = 0


@dataclass
class Vec3:
    x: float = 0
    y: float = 0
    z: float = 0


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
            parts: list[tuple[int, int, int]] = zip(face.pts, face.uvs, face.normals) # type: ignore
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

@dataclass
class Object:
    name: str = "Unnamed"
    mesh: Mesh = field(default_factory=Mesh)
    path: Path = Path()

    def save(self, dir: Path):
        self.path = dir / f"{self.name.replace('.', '_').replace('/', '_')}.obj"
        with open(self.path, "w") as f:
            f.write(f"o {self.name}\n")
            self.mesh.write(f)


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
                    parts2: list[list[int]] = [list(map(lambda s: int(s) - 1, p.split("/"))) for p in parts[1:]]
                    pts_i, uvs_i, normals_i = zip(*parts2)
                    face: Face = Face(pts_i, normals_i, uvs_i)
                    obj.mesh.faces.append(face)
            
            print(f"{len(pts)} pts, {len(normals)} normals, {len(uvs)} uvs")
            for obj in self.objects:
                obj.mesh.extract(pts, normals, uvs)


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
                        "texture": "generalTex.png"
                    }
                    for obj in self.objects
                ]
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)
            else:
                print(f"Metadata file not found (looked at {meta_path})")


if __name__ == "__main__":
    root: Path = Path(__file__).parent.parent
    splitter: MeshSplitter = MeshSplitter(root / "assets" / "SimpleTrack" / "Obstacles.obj", root / "assets" / "SimpleTrack" / "obstacles")
    splitter.split(True)
