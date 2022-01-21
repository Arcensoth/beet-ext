import logging
from pathlib import Path
from typing import Any, Iterable, List

from beet import Context, PackConfig, ProjectConfig, configurable, subproject
from beet.contrib.load import load
from beet.toolchain.cli import secho
from pydantic import BaseModel, Field, validator

log = logging.getLogger("subprojects")


class SubprojectsOptions(BaseModel):
    match: str
    merge: List[str] = Field(default_factory=list)
    config: ProjectConfig = Field(default_factory=ProjectConfig)

    @validator("merge", pre=True, allow_reuse=True)
    def split_merge(cls, value: Any):
        if isinstance(value, str):
            return value.split(",")
        return value


def beet_default(ctx: Context):
    ctx.require(subprojects)


@configurable(validator=SubprojectsOptions)
def subprojects(ctx: Context, opts: SubprojectsOptions):
    if opts.merge:
        merge_subprojects(ctx, opts)
    else:
        build_subprojects(ctx, opts)


def merge_subprojects(ctx: Context, opts: SubprojectsOptions):
    to_merge: List[str] = opts.merge.copy()
    merged: List[str] = []
    for root in resolve_subproject_roots(ctx, opts):
        if root.name in to_merge:
            merge_subproject(ctx, opts, root)
            to_merge.remove(root.name)
            merged.append(root.name)
    if to_merge:
        to_merge_str = ", ".join(to_merge)
        secho(
            f"Missing {len(to_merge)} of {len(opts.merge)} subprojects: {to_merge_str}",
            fg="yellow",
        )
    merged_str = ", ".join(merged)
    secho(
        f"Finished merging {len(merged)} of {len(opts.merge)} subprojects: {merged_str}",
        fg="green",
    )


def merge_subproject(ctx: Context, opts: SubprojectsOptions, root: Path):
    root_rel = root.relative_to(ctx.directory)
    secho(f"Merging subproject: {root_rel}")
    ctx.require(load(data_pack=[str(root)]))


def build_subprojects(ctx: Context, opts: SubprojectsOptions):
    built: List[str] = []
    for root in resolve_subproject_roots(ctx, opts):
        build_subproject(ctx, opts, root)
        built.append(root.name)
    secho(f"Finished building {len(built)} subprojects", fg="green")


def build_subproject(ctx: Context, opts: SubprojectsOptions, root: Path):
    root_rel = root.relative_to(ctx.directory)
    secho(f"Building subproject: {root_rel}")
    config = ProjectConfig(
        directory=str(root),
        data_pack=PackConfig(
            load=[str(root)],
        ),
        output=opts.config.output,
    ).with_defaults(opts.config)
    ctx.require(subproject(config))


def resolve_subproject_roots(ctx: Context, opts: SubprojectsOptions) -> Iterable[Path]:
    for node in ctx.directory.glob(opts.match):
        if node.is_dir():
            yield node
        elif node.is_file():
            yield node.parent
        else:
            node_rel = node.relative_to(ctx.directory)
            log.warning(f"Skipping unsupported subproject node: {node_rel}")
