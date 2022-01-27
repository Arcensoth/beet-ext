import logging
from pathlib import Path
from typing import Any, Iterable, List

from beet import Context, PackConfig, ProjectConfig, configurable, subproject
from beet.contrib.load import load
from pydantic import BaseModel, Field, validator

log = logging.getLogger("subprojects")


class SubprojectsOptions(BaseModel):
    root: Path
    match: str
    merge: List[Path] = Field(default_factory=list)
    config: ProjectConfig = Field(default_factory=ProjectConfig)

    @validator("root", pre=True, allow_reuse=True)
    def resolve_root(cls, value: Any):
        return Path(value).absolute()

    @validator("merge", pre=True, allow_reuse=True)
    def split_merge(cls, value: Any):
        if isinstance(value, str):
            return [Path(s) for s in value.split(",")]
        return value


def beet_default(ctx: Context):
    ctx.require(subprojects)


@configurable(validator=SubprojectsOptions)
def subprojects(ctx: Context, opts: SubprojectsOptions):
    opts.config.resolve(opts.root)
    if opts.merge:
        merge_subprojects(ctx, opts)
    else:
        build_subprojects(ctx, opts)


def merge_subprojects(ctx: Context, opts: SubprojectsOptions):
    missing: List[Path] = []
    merged: List[Path] = []
    for root_rel in opts.merge:
        root = opts.root / root_rel
        if not root.exists():
            missing.append(root_rel)
            continue
        merge_subproject(ctx, opts, root)
        merged.append(root_rel)
    if missing:
        to_merge_str = ", ".join(str(p) for p in missing)
        log.warning(
            f"Missing {len(missing)} of {len(opts.merge)} subprojects: {to_merge_str}"
        )
    merged_str = ", ".join(str(p) for p in merged)
    log.info(
        f"Finished merging {len(merged)} of {len(opts.merge)} subprojects: {merged_str}"
    )


def merge_subproject(ctx: Context, opts: SubprojectsOptions, root: Path):
    root_rel = root.relative_to(opts.root)
    log.info(f"Merging subproject: {root_rel}")
    ctx.require(load(data_pack=[str(root)]))


def build_subprojects(ctx: Context, opts: SubprojectsOptions):
    built: List[Path] = []
    for root in resolve_subproject_roots(ctx, opts):
        build_subproject(ctx, opts, root)
        built.append(root.relative_to(opts.root))
    log.info(f"Finished building {len(built)} subprojects")


def build_subproject(ctx: Context, opts: SubprojectsOptions, root: Path):
    root_rel = root.relative_to(opts.root)
    log.info(f"Building subproject: {root_rel}")
    config = ProjectConfig(
        directory=str(root),
        data_pack=PackConfig(
            load=[str(root)],
        ),
        output=opts.config.output,
    ).with_defaults(opts.config)
    ctx.require(subproject(config))


def resolve_subproject_roots(ctx: Context, opts: SubprojectsOptions) -> Iterable[Path]:
    for node in opts.root.glob(opts.match):
        if node.is_dir():
            yield node
        elif node.is_file():
            yield node.parent
        else:
            node_rel = node.relative_to(opts.root)
            log.warning(f"Skipping unsupported subproject node: {node_rel}")
