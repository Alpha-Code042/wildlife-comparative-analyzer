
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get(
        "SECRET_KEY",
        "change-this-development-key"
    )

    SQLALCHEMY_DATABASE_URI = (
        "sqlite:///" + os.path.join(
            BASE_DIR,
            "database",
            "wildlife.db"
        )
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB