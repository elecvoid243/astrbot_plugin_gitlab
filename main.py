from astrbot.api.star import Context, Star


class GitLabPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        from .adapter import GitLabPlatform

        GitLabPlatform.context = context
