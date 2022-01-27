import logging

import yaml
from beet import Context

log = logging.getLogger("subprojects")


def beet_default(ctx: Context):
    module_meta_file = ctx.directory / "module.yaml"

    with open(module_meta_file) as fp:
        module_meta = yaml.safe_load(fp)

    module_name = module_meta["name"]
    log.info(f"  -> name: {module_name}")

    module_version = module_meta["version"]
    log.info(f"  -> version: {module_version}")

    module_description = module_meta.get("description", "")
    log.info(f"  -> description: {module_description}")

    pack_title = module_meta.get("title") or module_name
    pack_description = [pack_title]
    if module_description:
        pack_description += [
            "\\n",
            {"text": "", "color": "light_gray", "extra": [module_description]},
        ]

    ctx.meta["module"] = module_meta

    ctx.project_description = pack_description
