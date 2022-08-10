from envpool.registration import register

register(
    task_id = "football",
    import_path = "envpool.gfootball",
    spec_cls = "FootballEnvSpec",
    env_cls = "FootballEnvpool",
)
