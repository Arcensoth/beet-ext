import logging

from beet import Context

log = logging.getLogger(__name__)


def beet_default(ctx: Context):
    livereload_logger = logging.getLogger("livereload")
    livereload_logger.addFilter(
        lambda r: not r.getMessage().startswith(
            "Creating a MIN function between two non-overlapping inputs"
        )
    )

    for processor_list in ctx.data.processor_lists.values():
        for processor in processor_list.data["processors"]:
            if processor["processor_type"] == "minecraft:rule":
                for rule in processor["rules"]:
                    if "location_predicate" not in rule:
                        rule["location_predicate"] = {
                            "predicate_type": "minecraft:always_true"
                        }

    for name, template_pool in ctx.data.template_pools.items():
        if template_pool.data["name"] != name:
            log.warning(f"Correcting template pool name mismatch in: {name}")
            template_pool.data["name"] = name
