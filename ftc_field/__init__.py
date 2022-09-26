from gym.envs.registration import register

register(
    id="ftc_field/ftc_field-v0",
    entry_point="ftc_field.envs:FieldEnvFTC",
)
