from beet import Context
from mecha import Mecha
from mecha.contrib.bolt import Runtime


def beet_default(ctx: Context):
    mc = ctx.inject(Mecha)
    runtime = ctx.inject(Runtime)
    runtime.expose(
        "raw", lambda cmd: runtime.commands.append(mc.parse(cmd, using="command"))
    )
