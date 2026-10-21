
import os

from sqlalchemy import inspect, text

from app import create_app, db
from app.models import (
    User,
    Media,
    ProcessingJob,
    AlgorithmBenchmark
)


app = create_app()


with app.app_context():
    # Ensure required directories exist.
    os.makedirs(
        os.path.join(app.root_path, "..", "database"),
        exist_ok=True
    )

    os.makedirs(
        app.config["UPLOAD_FOLDER"],
        exist_ok=True
    )

    # Create tables that do not already exist.
    db.create_all()

    # Add user_id to older media tables if necessary.
    inspector = inspect(db.engine)

    if "media" in inspector.get_table_names():
        columns = {
            column["name"]
            for column in inspector.get_columns("media")
        }

        if "user_id" not in columns:
            with db.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE media "
                        "ADD COLUMN user_id INTEGER "
                        "REFERENCES users(id)"
                    )
                )

            print("Database updated: media.user_id added.")


if __name__ == "__main__":
    app.run(debug=True)