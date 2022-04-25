from beet import Context
from bolt import Runtime
from mecha import Mecha


def beet_default(ctx: Context):
    mc = ctx.inject(Mecha)
    runtime = ctx.inject(Runtime)
    runtime.expose(
        "raw", lambda cmd: runtime.commands.append(mc.parse(cmd, using="command"))
    )
