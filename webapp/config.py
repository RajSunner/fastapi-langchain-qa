from pydantic import BaseSettings


class Settings(BaseSettings):
    emailw: str
    passwordw: str
    urlw: str
    okey: str

    class Config:
        env_file = ".env"
