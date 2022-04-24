# FIXME beet and 1.19 changes

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Annotated, Any, ClassVar, Iterable, Literal, TypeVar

from beet import Cache, Context, DataPack, Structure
from nbtlib import Compound
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class ExtendedJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, set):
            return list(o)
        if isinstance(o, PurePath):
            return str(o)
        if isinstance(o, BaseModel):
            if o.__custom_root_type__:
                return dict(getattr(o, "__root__"))
            return dict(o)
        return super().default(o)


class PoolDependents(BaseModel):
    features: set[str] = Field(default_factory=set)
    structures: set[str] = Field(default_factory=set)


class StructureDependents(BaseModel):
    pools: set[str] = Field(default_factory=set)


class Dependents(BaseModel):
    pool_dependents: dict[str, PoolDependents] = Field(default_factory=dict)
    structure_dependents: dict[str, StructureDependents] = Field(default_factory=dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_dependents = defaultdict(PoolDependents) | self.pool_dependents
        self.structure_dependents = (
            defaultdict(StructureDependents) | self.structure_dependents
        )


class FeatureNoSuchStartPool(BaseModel):
    type: Literal["feature.no_such_start_pool"] = "feature.no_such_start_pool"

    feature: str
    start_pool: str

    def __str__(self) -> str:
        return f"Unknown start pool `{self.start_pool}`"


class PoolNoSuchStructure(BaseModel):
    type: Literal["pool.no_such_structure"] = "pool.no_such_structure"

    pool: str
    structure: str

    def __str__(self) -> str:
        return f"Unknown structure `{self.structure}`"


class PoolNoSuchFallbackPool(BaseModel):
    type: Literal["pool.no_such_fallback_pool"] = "pool.no_such_fallback_pool"

    pool: str
    fallback_pool: str

    def __str__(self) -> str:
        return f"Unknown fallback pool `{self.fallback_pool}`"


class StructureNoSuchTargetPool(BaseModel):
    type: Literal["structure.no_such_target_pool"] = "structure.no_such_target_pool"

    structure: str
    position: tuple[int, int, int]
    target_pool: str

    def __str__(self) -> str:
        return (
            f"Unknown target pool `{self.target_pool}`"
            + f" in jigsaw block at {self.position}"
        )


class StructureNoSuchTargetStructure(BaseModel):
    type: Literal[
        "structure.no_such_target_structure"
    ] = "structure.no_such_target_structure"

    structure: str
    position: tuple[int, int, int]
    target_pool: str
    target_structure: str

    def __str__(self) -> str:
        return (
            f"Unknown target structure `{self.target_structure}`"
            + f" in target pool `{self.target_pool}`"
            + f" in jigsaw block at {self.position}"
        )


class StructureNoSuchFallbackPool(BaseModel):
    type: Literal["structure.no_such_fallback_pool"] = "structure.no_such_fallback_pool"

    structure: str
    position: tuple[int, int, int]
    target_pool: str
    fallback_pool: str

    def __str__(self) -> str:
        return (
            f"Unknown fallback pool `{self.fallback_pool}`"
            + f" in target pool `{self.target_pool}`"
            + f" in jigsaw block at {self.position}"
        )


class StructureNoSuchFallbackStructure(BaseModel):
    type: Literal[
        "structure.no_such_fallback_structure"
    ] = "structure.no_such_fallback_structure"

    structure: str
    position: tuple[int, int, int]
    target_pool: str
    fallback_pool: str
    fallback_structure: str

    def __str__(self) -> str:
        return (
            f"Unknown fallback structure `{self.fallback_structure}`"
            + f" in fallback pool `{self.fallback_pool}`"
            + f" in target pool `{self.target_pool}`"
            + f" in jigsaw block at {self.position}"
        )


Problem = (
    FeatureNoSuchStartPool
    | PoolNoSuchStructure
    | PoolNoSuchFallbackPool
    | StructureNoSuchTargetPool
    | StructureNoSuchTargetStructure
    | StructureNoSuchFallbackPool
    | StructureNoSuchFallbackStructure
)


class NodeBase(BaseModel):
    type: Literal["node"] = "node"
    name: str

    path: PurePath | None = None
    old_path: PurePath | None = None

    processed: bool = False

    dependencies: set[str] = Field(default_factory=set)
    dependents: set[str] = Field(default_factory=set)
    problems: list[Annotated[Problem, Field(discriminator="type")]] = Field(
        default_factory=list
    )

    key_prefix: ClassVar[str] = "node"

    def __str__(self) -> str:
        return f"{self.type} `{self.name}`"

    @property
    def key(self) -> str:
        return f"{self.type}@{self.name}"

    @property
    def is_new(self) -> bool:
        return (self.path is not None) and (self.old_path is None)

    @property
    def was_deleted(self) -> bool:
        return (self.path is None) and (self.old_path is not None)

    def clean_dependencies(self, jig: "JigsawValidator"):
        # TODO Only update dependencies when they are added/removed. #optimize
        for dep_key in self.dependencies:
            if (dep_node := jig.store.nodes.get(dep_key)) and (
                self.key in dep_node.dependents
            ):
                dep_node.dependents.remove(self.key)
        self.dependencies.clear()

    def add_pool_dependency(self, jig: "JigsawValidator", pool_name: str):
        pool_key, pool_node = jig.store.get_pool(pool_name)
        self.dependencies.add(pool_key)
        pool_node.dependents.add(self.key)

    def add_structure_dependency(self, jig: "JigsawValidator", structure_name: str):
        structure_key, structure_node = jig.store.get_structure(structure_name)
        self.dependencies.add(structure_key)
        structure_node.dependents.add(self.key)

    def touch(self):
        self.processed = False
        self.old_path = self.path
        self.path = None

    def process(self, jig: "JigsawValidator"):
        log.debug(f"Processing {self}")
        self.processed = True
        self.problems.clear()
        self.clean_dependencies(jig)
        self.verify(jig)

    def verify(self, jig: "JigsawValidator"):
        ...


class FeatureNode(NodeBase):
    type: Literal["feature"] = "feature"
    key_prefix = "feature"

    def verify(self, jig: "JigsawValidator"):
        feature = jig.data.worldgen_structures[self.name]

        # Make sure the start pool exists.
        if (config := feature.data.get("config")) and (
            start_pool_name := config.get("start_pool")
        ):
            self.add_pool_dependency(jig, start_pool_name)
            if start_pool_name not in jig.data.template_pools:
                self.problems.append(
                    FeatureNoSuchStartPool(
                        feature=self.name, start_pool=start_pool_name
                    )
                )


class PoolNode(NodeBase):
    type: Literal["pool"] = "pool"
    key_prefix = "pool"

    def verify(self, jig: "JigsawValidator"):
        pool = jig.data.template_pools[self.name]

        # Make sure the fallback pool exists.
        if (fallback_pool_name := pool.data.get("fallback")) and (
            fallback_pool_name != "minecraft:empty"
        ):
            self.add_pool_dependency(jig, fallback_pool_name)
            if fallback_pool_name not in jig.data.template_pools:
                self.problems.append(
                    PoolNoSuchFallbackPool(
                        pool=self.name, fallback_pool=fallback_pool_name
                    )
                )

        # Make sure each structure element exists.
        for element_entry in pool.data["elements"]:
            element = element_entry["element"]
            element_type = element["element_type"]
            if element_type == "minecraft:single_pool_element":
                structure_name = element["location"]
                self.add_structure_dependency(jig, structure_name)
                if structure_name not in jig.data.structures:
                    self.problems.append(
                        PoolNoSuchStructure(pool=self.name, structure=structure_name)
                    )


class StructureNode(NodeBase):
    type: Literal["structure"] = "structure"
    key_prefix = "structure"

    @classmethod
    def iter_structure_elements(cls, pool: TemplatePool) -> Iterable[dict[str, Any]]:
        for element_entry in pool.data["elements"]:
            element = element_entry["element"]
            element_type = element["element_type"]
            if element_type == "minecraft:single_pool_element":
                yield element

    @classmethod
    def iter_jigsaw_blocks(cls, structure: Structure) -> Iterable[Compound]:
        for block in structure.data["blocks"]:
            if (block_nbt := block.get("nbt")) and (
                block_nbt.get("id") == "minecraft:jigsaw"
            ):
                yield block

    @classmethod
    def get_structure_size(cls, structure: Structure) -> tuple[int, int, int]:
        size: Any = structure.data.get("size")
        try:
            return tuple(int(n) for n in size)
        except:
            log.exception(f"Invalid size: {size}")
        return (0, 0, 0)

    def get_structure(self, jig: "JigsawValidator") -> Structure:
        return jig.data.structures[self.name]

    def verify(self, jig: "JigsawValidator"):
        structure = self.get_structure(jig)

        # Look for jigsaw blocks in the struture.
        for jigsaw_block in self.iter_jigsaw_blocks(structure):
            self.verify_jigsaw_block(jig, jigsaw_block)

    def verify_jigsaw_block(self, jig: "JigsawValidator", jigsaw_block: Compound):
        target_pool_name = str(jigsaw_block["nbt"].get("pool"))

        # Continue only if the jigsaw block has a non-empty target pool.
        if target_pool_name == "minecraft:empty":
            return

        # Register the target pool as a dependency.
        self.add_pool_dependency(jig, target_pool_name)

        position = tuple(int(c) for c in jigsaw_block["pos"])

        # If the target pool exists, further verify it.
        if target_pool := jig.data.template_pools.get(target_pool_name):
            self.verify_target_pool(jig, position, target_pool_name, target_pool)

        # If the target pool does not exist, we have a problem.
        else:
            self.problems.append(
                StructureNoSuchTargetPool(
                    structure=self.name,
                    position=position,
                    target_pool=target_pool_name,
                )
            )

    def verify_target_pool(
        self,
        jig: "JigsawValidator",
        position: tuple[int, int, int],
        target_pool_name: str,
        target_pool: TemplatePool,
    ):
        # Verify each target structure in the target pool.
        for target_element in self.iter_structure_elements(target_pool):
            target_structure_name = target_element["location"]

            # Register the target structure as a dependency.
            self.add_structure_dependency(jig, target_structure_name)

            # If the target structure exists, further verify it.
            if target_structure := jig.data.structures.get(target_structure_name):
                self.verify_target_structure(
                    jig,
                    position,
                    target_pool_name,
                    target_pool,
                    target_structure_name,
                    target_structure,
                )

            # If the target structure does not exist, we have a problem.
            else:
                self.problems.append(
                    StructureNoSuchTargetStructure(
                        structure=self.name,
                        position=position,
                        target_pool=target_pool_name,
                        target_structure=target_structure_name,
                    )
                )

        # Check if there is a non-empty fallback pool.
        if (fallback_pool_name := target_pool.data.get("fallback")) and (
            fallback_pool_name != "minecraft:empty"
        ):
            # Register the fallback pool as a dependency.
            self.add_pool_dependency(jig, fallback_pool_name)

            # If the fallback pool exists, verify it further.
            if fallback_pool := jig.data.template_pools.get(fallback_pool_name):
                self.verify_fallback_pool(
                    jig,
                    target_pool_name,
                    target_pool,
                    position,
                    fallback_pool_name,
                    fallback_pool,
                )

            # If the fallback pool does not exist, we have a problem.
            else:
                self.problems.append(
                    StructureNoSuchFallbackPool(
                        structure=self.name,
                        position=position,
                        target_pool=target_pool_name,
                        fallback_pool=fallback_pool_name,
                    )
                )

    def verify_fallback_pool(
        self,
        jig: "JigsawValidator",
        target_pool_name: str,
        target_pool: TemplatePool,
        position: tuple[int, int, int],
        fallback_pool_name: str,
        fallback_pool: TemplatePool,
    ):
        # Verify each fallback structure in the fallback pool.
        for structure_element in self.iter_structure_elements(fallback_pool):
            fallback_structure_name = structure_element["location"]

            # Register the fallback structure as a dependency.
            self.add_structure_dependency(jig, fallback_structure_name)

            # If the fallback structure exists, verify it further.
            if fallback_structure := jig.data.structures.get(fallback_structure_name):
                self.verify_fallback_structure(
                    jig,
                    position,
                    target_pool_name,
                    target_pool,
                    fallback_structure_name,
                    fallback_structure,
                )

            # If the fallback structure does not exist, we have a problem.
            else:
                self.problems.append(
                    StructureNoSuchFallbackStructure(
                        structure=self.name,
                        position=position,
                        target_pool=target_pool_name,
                        fallback_pool=fallback_pool_name,
                        fallback_structure=fallback_structure_name,
                    )
                )

    def verify_target_structure(
        self,
        jig: "JigsawValidator",
        position: tuple[int, int, int],
        target_pool_name: str,
        target_pool: TemplatePool,
        target_structure_name: str,
        target_structure: Structure,
    ):
        # IMPL Verify that the target structure does not collide.
        # TODO Account for structures with non-zero origins (external tools).
        parent_structure = self.get_structure(jig)
        parent_size = self.get_structure_size(parent_structure)
        child_size = self.get_structure_size(target_structure)
        parent_c1 = (0, 0, 0)
        parent_c2 = tuple(parent_c1[i] + parent_size[i] for i in range(3))
        child_c1 = position
        child_c2 = tuple(child_c1[i] + child_size[i] for i in range(3))
        log.info(
            f"  parent box from {parent_c1} to {parent_c2}"
            + f", child box from {child_c1} to {child_c2}"
        )

    def verify_fallback_structure(
        self,
        jig: "JigsawValidator",
        position: tuple[int, int, int],
        target_pool_name: str,
        target_pool: TemplatePool,
        fallback_structure_name: str,
        fallback_structure: Structure,
    ):
        # IMPL Verify that the fallback structure does not collide.
        ...


Node = FeatureNode | PoolNode | StructureNode
NT = TypeVar("NT", bound=Node)


class Store(BaseModel):
    nodes: dict[str, Annotated[Node, Field(discriminator="type")]] = Field(
        default_factory=dict
    )

    def _get(
        self, name: str, node_type: type[NT], *, path: PurePath | None = None
    ) -> tuple[str, NT]:
        key = f"{node_type.key_prefix}@{name}"
        node = self.nodes.get(key)
        if node is None:
            node = node_type(name=name)
            self.nodes[key] = node
        elif not isinstance(node, node_type):
            raise ValueError(
                f"Expected node type `{node_type.__name__}`"
                + f" but got `{type(node).__name__}`"
            )
        if path:
            node.path = path
        return key, node

    def get_feature(
        self, name: str, *, path: PurePath | None = None
    ) -> tuple[str, FeatureNode]:
        return self._get(name, FeatureNode, path=path)

    def get_pool(
        self, name: str, *, path: PurePath | None = None
    ) -> tuple[str, PoolNode]:
        return self._get(name, PoolNode, path=path)

    def get_structure(
        self, name: str, *, path: PurePath | None = None
    ) -> tuple[str, StructureNode]:
        return self._get(name, StructureNode, path=path)

    def iter_unprocessed_dependents(self, node: Node) -> Iterable[Node]:
        for dep_key in node.dependents:
            dep_node = self.nodes.get(dep_key)
            if dep_node and not dep_node.processed:
                yield dep_node

    def iter_removed_nodes(self) -> Iterable[Node]:
        yield from (node for node in self.nodes.values() if node.was_deleted)

    def iter_added_nodes(self) -> Iterable[Node]:
        yield from (node for node in self.nodes.values() if node.is_new)


@dataclass
class JigsawValidator:
    ctx: Context

    data: DataPack = field(init=False)
    cache: Cache = field(init=False)
    store: Store = field(init=False)

    def __post_init__(self):
        self.data = self.ctx.data
        self.cache = self.ctx.cache["jigsaw"]
        self.store = self.load_store()

    @property
    def store_path(self) -> Path:
        return self.cache.directory / "store.json"

    def load_store(self) -> Store:
        try:
            return Store.parse_file(self.store_path)
        except:
            return Store()

    def dump_store(self):
        with self.store_path.open("w") as fp:
            json.dump(self.store, fp, cls=ExtendedJsonEncoder, indent=2)

    def run(self):
        log.info("Starting jigsaw validation...")

        for node in self.store.nodes.values():
            node.touch()

        for name, feature in self.data.worldgen_structures.items():
            path = feature.original.ensure_source_path()
            node_path = PurePath(path).relative_to(self.ctx.directory)
            key, node = self.store.get_feature(name, path=node_path)
            if self.cache.has_changed(path):
                log.info(f"Detected CHANGED {node}")
                node.process(self)

        for name, pool in self.data.template_pools.items():
            path = pool.original.ensure_source_path()
            node_path = PurePath(path).relative_to(self.ctx.directory)
            key, node = self.store.get_pool(name, path=node_path)
            if self.cache.has_changed(path):
                log.info(f"Detected CHANGED {node}")
                node.process(self)

        for name, structure in self.data.structures.items():
            path = structure.original.ensure_source_path()
            node_path = PurePath(path).relative_to(self.ctx.directory)
            key, node = self.store.get_structure(name, path=node_path)
            if self.cache.has_changed(path):
                log.info(f"Detected CHANGED {node}")
                node.process(self)

        for node in self.store.iter_added_nodes():
            log.info(f"Detected NEW {node}")
            for node in self.store.iter_unprocessed_dependents(node):
                node.process(self)

        for node in self.store.iter_removed_nodes():
            log.info(f"Detected REMOVED {node}")
            for node in self.store.iter_unprocessed_dependents(node):
                node.process(self)

        for node in self.store.nodes.values():
            if not node.path:
                continue
            if len(node.problems) == 1:
                log.warning(str(node.problems[0]), extra={"annotate": node.path})
            elif node.problems:
                log.warning(
                    f"Found {len(node.problems)} problems with {node}:\n- "
                    + "\n- ".join(str(p) for p in node.problems),
                    extra={"annotate": node.path},
                )

        self.dump_store()

        log.info("Finished jigsaw validation!")


def beet_default(ctx: Context):
    jig = JigsawValidator(ctx)
    jig.run()
