import logging

from beet import Context


def beet_default(ctx: Context):
    logger = logging.getLogger("livereload")
    logger.addFilter(
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
