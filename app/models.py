
from app import db
from flask_login import UserMixin


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    processing_jobs = db.relationship(
        "ProcessingJob",
        backref="user",
        lazy=True
    )

    media_files = db.relationship(
        "Media",
        backref="user",
        lazy=True
    )


class Media(db.Model):
    __tablename__ = "media"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=True
    )

    filename = db.Column(
        db.String(255),
        nullable=False
    )

    file_type = db.Column(
        db.String(50),
        nullable=False
    )

    file_path = db.Column(
        db.String(500),
        nullable=False
    )

    uploaded_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    )

    processing_jobs = db.relationship(
        "ProcessingJob",
        backref="media",
        lazy=True
    )


class ProcessingJob(db.Model):
    __tablename__ = "processing_jobs"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    media_id = db.Column(
        db.Integer,
        db.ForeignKey("media.id"),
        nullable=False
    )

    algorithm = db.Column(
        db.String(100),
        nullable=False
    )

    status = db.Column(
        db.String(50),
        default="pending"
    )

    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)


class AlgorithmBenchmark(db.Model):
    __tablename__ = "algorithm_benchmarks"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    processing_job_id = db.Column(
        db.Integer,
        db.ForeignKey("processing_jobs.id"),
        nullable=False
    )

    algorithm = db.Column(
        db.String(100),
        nullable=False
    )

    processing_time = db.Column(db.Float)
    fps = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    map_score = db.Column(db.Float)
    mota = db.Column(db.Float)
    motp = db.Column(db.Float)

    processing_job = db.relationship(
        "ProcessingJob",
        backref=db.backref(
            "benchmark",
            uselist=False
        )
    )