import json
from typing import Any, Iterable

from beet import Context, configurable
from mecha import Mecha
from mecha.contrib.bolt import Runtime
from pydantic import BaseModel, Field


class LogChannel(BaseModel):
    label: Any
    color: str


class LogOptions(BaseModel):
    storage_location: str = "beet_ext:log"
    text_as_server_component: Any = {"text": "Server", "color": "dark_purple"}
    text_as_self_component = {"selector": "@s"}
    enabled: list[str] = Field(default_factory=list)
    channels: dict[str, LogChannel]

    @property
    def text_as_server_json(self) -> str:
        return json.dumps(self.text_as_server_component)

    @property
    def text_as_self_json(self) -> str:
        return json.dumps(self.text_as_self_component)

    def skip_channel(self, name: str) -> bool:
        return bool(self.enabled) and (name not in self.enabled)

    def get_channel(self, name: str) -> LogChannel:
        channel = self.channels.get(name, DEFAULT_CHANNELS.get(name))
        if not channel:
            raise KeyError(f"No such log channel: {name}")
        return channel


def beet_default(ctx: Context):
    ctx.require(log)


@configurable(validator=LogOptions)
def log(ctx: Context, opts: LogOptions):
    mc = ctx.inject(Mecha)
    runtime = ctx.inject(Runtime)

    def log(message: Any, channel: str = "info"):
        if opts.skip_channel(channel):
            return
        for cmd in iter_log_commands(
            runtime.get_path(), opts, message, opts.get_channel(channel)
        ):
            runtime.commands.append(mc.parse(cmd, using="command"))

    runtime.expose("log", log)


DEFAULT_CHANNELS: dict[str, LogChannel] = {
    "debug": LogChannel(
        label=[{"text": "  ", "bold": True}, "DEBUG", {"text": "  ", "bold": True}],
        color="aqua",
    ),
    "info": LogChannel(
        label=[{"text": "  ", "bold": True}, " INFO ", {"text": "  ", "bold": True}],
        color="green",
    ),
    "warning": LogChannel(
        label=[{"text": " ", "bold": True}, "WARNING", {"text": " ", "bold": True}],
        color="yellow",
    ),
    "error": LogChannel(
        label=[{"text": "  ", "bold": True}, "ERROR", {"text": "  ", "bold": True}],
        color="red",
    ),
    "critical": LogChannel(
        label=[
            {"text": " ", "color": "dark_gray"},
            "CRITICAL",
            {"text": ".", "color": "dark_gray"},
        ],
        color="light_purple",
    ),
}


def iter_log_commands(
    path: str, opts: LogOptions, message: Any, channel: LogChannel
) -> Iterable[str]:
    yield (
        f"execute store result storage {opts.storage_location} gametime int 1.0 "
        " run time query gametime"
    )
    yield (
        f"data modify storage {opts.storage_location} text.as"
        f" set value {opts.text_as_server_json}"
    )
    yield (
        "execute if entity @s"
        f" run data modify storage {opts.storage_location} text.as"
        f" set value {opts.text_as_self_json}"
    )
    log_json = build_log_json(path, opts, message, channel)
    yield f"tellraw @a {log_json}"


def build_log_json(
    path: str, opts: LogOptions, message: Any, channel: LogChannel
) -> str:
    namespace = path.split(":")[0]
    return json.dumps(
        [
            {"text": "", "color": channel.color},
            {
                "text": "",
                "extra": [
                    {"text": "[", "color": "#aaaaaa"},
                    channel.label,
                    {"text": "]", "color": "#aaaaaa"},
                    " ",
                    {
                        "storage": opts.storage_location,
                        "path": "gametime",
                        "color": "#777777",
                    },
                    " ",
                    {"text": f"{namespace}", "color": "#aaaaaa"},
                ],
                "hoverEvent": {
                    "action": "show_text",
                    "contents": [
                        "",
                        {"text": "in ", "color": "#777777"},
                        {"text": f"{path}", "color": "yellow"},
                        "\n",
                        {"text": "as ", "color": "#777777"},
                        {
                            "text": "",
                            "color": "aqua",
                            "extra": [
                                {
                                    "storage": opts.storage_location,
                                    "nbt": "text.as",
                                    "interpret": True,
                                }
                            ],
                        },
                    ],
                },
            },
            "  ",
            message,
        ]
    )
