import os
import uuid
import time
from pathlib import Path

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    current_app,
    abort,
    send_from_directory,
)

from flask_login import (
    login_user,
    logout_user,
    login_required,
    current_user,
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from werkzeug.utils import secure_filename

from app import db
from app.models import (
    User,
    Media,
    ProcessingJob,
    AlgorithmBenchmark,
)

from algorithms.classical_processor import analyze_classical_image
from algorithms.image_processor import analyze_image
from algorithms.video_processor import analyze_video
from algorithms.video_feature_processor import analyze_video_features
from algorithms.sift_processor import analyze_sift_image
from algorithms.hog_processor import analyze_hog_image


main = Blueprint("main", __name__)


ALLOWED_IMAGE_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp",
}

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4", "avi", "mov", "mkv", "webm",
}


def get_file_type(filename):
    """Identify supported image and video files."""

    if "." not in filename:
        return None

    extension = filename.rsplit(".", 1)[1].lower()

    if extension in ALLOWED_IMAGE_EXTENSIONS:
        return "image"

    if extension in ALLOWED_VIDEO_EXTENSIONS:
        return "video"

    return None


def get_safe_media_path(media):
    """Verify that a media file is inside the upload directory."""

    upload_folder = os.path.realpath(
        current_app.config["UPLOAD_FOLDER"]
    )

    stored_path = os.path.realpath(media.file_path)

    try:
        if os.path.commonpath(
            [upload_folder, stored_path]
        ) != upload_folder:
            abort(404)
    except ValueError:
        abort(404)

    if not os.path.isfile(stored_path):
        abort(404)

    return upload_folder, stored_path


def get_detection_result_path(media):
    """Build and validate the expected annotated image path."""

    project_dir = Path(current_app.root_path).parent

    results_folder = os.path.realpath(
        project_dir / "results" / "detections"
    )

    stored_path = os.path.realpath(media.file_path)

    output_filename = Path(stored_path).stem + "_detected.jpg"

    output_path = os.path.realpath(
        os.path.join(results_folder, output_filename)
    )

    try:
        if os.path.commonpath(
            [results_folder, output_path]
        ) != results_folder:
            abort(404)
    except ValueError:
        abort(404)

    if not os.path.isfile(output_path):
        abort(404)

    return results_folder, output_filename


def get_video_result_path(media):
    """Build and validate the expected processed-video path."""

    project_dir = Path(current_app.root_path).parent

    results_folder = os.path.realpath(
        project_dir / "results" / "videos"
    )

    stored_path = os.path.realpath(media.file_path)

    output_filename = Path(stored_path).stem + "_detected.mp4"

    output_path = os.path.realpath(
        os.path.join(results_folder, output_filename)
    )

    try:
        if os.path.commonpath(
            [results_folder, output_path]
        ) != results_folder:
            abort(404)
    except ValueError:
        abort(404)

    if not os.path.isfile(output_path):
        abort(404)

    return results_folder, output_filename


def get_safe_deletion_paths(media):
    """Return validated uploaded and generated file paths."""

    upload_folder = os.path.realpath(
        current_app.config["UPLOAD_FOLDER"]
    )

    stored_path = os.path.realpath(media.file_path)

    try:
        if os.path.commonpath(
            [upload_folder, stored_path]
        ) != upload_folder:
            abort(404)
    except ValueError:
        abort(404)

    project_dir = Path(current_app.root_path).parent
    media_stem = Path(stored_path).stem

    safe_paths = [stored_path]

    result_definitions = {
        "detections": [
            f"{media_stem}_detected.jpg",
        ],
        "videos": [
            f"{media_stem}_detected.mp4",
            f"{media_stem}_edges.mp4",
            f"{media_stem}_sift_features.mp4",
            f"{media_stem}_hog_features.mp4",
        ],
        "sift": [
            f"{media_stem}_sift_features.jpg",
        ],
        "hog": [
            f"{media_stem}_hog_features.jpg",
        ],
        "classical": [
            f"{media_stem}_grayscale.jpg",
            f"{media_stem}_edges.jpg",
        ],
    }

    for folder_name, filenames in result_definitions.items():
        results_folder = os.path.realpath(
            project_dir / "results" / folder_name
        )

        for filename in filenames:
            output_path = os.path.realpath(
                os.path.join(results_folder, filename)
            )

            try:
                if os.path.commonpath(
                    [results_folder, output_path]
                ) != results_folder:
                    abort(404)
            except ValueError:
                abort(404)

            safe_paths.append(output_path)

    return safe_paths


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if not username or not email or not password:
            flash("Please complete all fields.", "error")
            return redirect(url_for("main.register"))

        if User.query.filter_by(email=email).first():
            flash("This email is already registered.", "error")
            return redirect(url_for("main.register"))

        if User.query.filter_by(username=username).first():
            flash("This username is already taken.", "error")
            return redirect(url_for("main.register"))

        user = User(
            username=username,
            email=email,
            password=generate_password_hash(password),
        )

        db.session.add(user)
        db.session.commit()

        flash(
            "Account created successfully. You can now log in.",
            "success",
        )

        return redirect(url_for("main.login"))

    return render_template("register.html")


@main.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)

            flash(
                f"Welcome back, {user.username}!",
                "success",
            )

            return redirect(url_for("main.dashboard"))

        flash(
            "Invalid email or password. Please try again.",
            "error",
        )

        return redirect(url_for("main.login"))

    return render_template("login.html")


@main.route("/dashboard")
@login_required
def dashboard():
    media_files = Media.query.filter_by(
        user_id=current_user.id
    ).order_by(Media.uploaded_at.desc()).all()

    return render_template(
        "dashboard.html",
        media_files=media_files,
    )


@main.route("/upload", methods=["GET", "POST"])
@login_required
def upload_media():
    if request.method == "POST":
        if "media_file" not in request.files:
            flash("Please select a file to upload.", "error")
            return redirect(url_for("main.upload_media"))

        file = request.files["media_file"]

        if not file.filename or not file.filename.strip():
            flash("Please select a file to upload.", "error")
            return redirect(url_for("main.upload_media"))

        original_filename = secure_filename(file.filename)

        if not original_filename:
            flash("The selected filename is invalid.", "error")
            return redirect(url_for("main.upload_media"))

        file_type = get_file_type(original_filename)

        if file_type is None:
            flash(
                "Unsupported file type. Please select a supported image or video.",
                "error",
            )
            return redirect(url_for("main.upload_media"))

        extension = original_filename.rsplit(".", 1)[1].lower()
        stored_filename = f"{uuid.uuid4().hex}.{extension}"

        upload_folder = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, stored_filename)

        try:
            file.save(file_path)

            media = Media(
                user_id=current_user.id,
                filename=original_filename,
                file_type=file_type,
                file_path=file_path,
            )

            db.session.add(media)
            db.session.commit()

        except Exception:
            db.session.rollback()

            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    current_app.logger.exception(
                        "Could not remove failed upload: %s",
                        file_path,
                    )

            current_app.logger.exception("Media upload failed.")

            flash(
                "The upload could not be completed. Please try again.",
                "error",
            )

            return redirect(url_for("main.upload_media"))

        flash(
            f"{original_filename} uploaded successfully.",
            "success",
        )

        return redirect(url_for("main.dashboard"))

    return render_template("upload.html")


@main.route("/media/<int:media_id>")
@login_required
def view_media(media_id):
    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    upload_folder, stored_path = get_safe_media_path(media)

    response = send_from_directory(
        upload_folder,
        os.path.basename(stored_path),
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@main.route("/delete-media/<int:media_id>", methods=["POST"])
@login_required
def delete_media(media_id):
    """Delete owned media and its related analysis records."""

    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    deletion_paths = get_safe_deletion_paths(media)

    try:
        jobs = ProcessingJob.query.filter_by(
            media_id=media.id
        ).all()

        job_ids = [job.id for job in jobs]

        if job_ids:
            AlgorithmBenchmark.query.filter(
                AlgorithmBenchmark.processing_job_id.in_(job_ids)
            ).delete(synchronize_session=False)

            for job in jobs:
                db.session.delete(job)

        db.session.delete(media)
        db.session.commit()

    except Exception:
        db.session.rollback()

        current_app.logger.exception(
            "Could not delete media record %s.",
            media_id,
        )

        flash(
            "The media could not be deleted. Please try again.",
            "error",
        )

        return redirect(url_for("main.dashboard"))

    file_cleanup_failed = False

    for path in deletion_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            file_cleanup_failed = True

            current_app.logger.exception(
                "Database record was deleted, but file cleanup failed: %s",
                path,
            )

    if file_cleanup_failed:
        flash(
            "The media record was deleted, but a file could not be removed "
            "from storage. Check the application terminal.",
            "error",
        )
    else:
        flash("Media deleted successfully.", "success")

    return redirect(url_for("main.dashboard"))


@main.route("/compare/<int:media_id>")
@login_required
def compare_image(media_id):
    """Compare YOLO, classical OpenCV, SIFT, and HOG."""

    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    if media.file_type != "image":
        flash(
            "Image comparison is available for images only.",
            "error",
        )
        return redirect(url_for("main.dashboard"))

    try:
        _, stored_path = get_safe_media_path(media)

        yolo_start = time.perf_counter()
        yolo_analysis = analyze_image(stored_path)
        yolo_time = round(time.perf_counter() - yolo_start, 4)

        classical_analysis = analyze_classical_image(stored_path)
        sift_analysis = analyze_sift_image(stored_path)
        hog_analysis = analyze_hog_image(stored_path)

        yolo_image_url = url_for(
            "main.view_analysis_result",
            media_id=media.id,
        )

        grayscale_url = url_for(
            "main.view_classical_result",
            media_id=media.id,
            result_type="grayscale",
        )

        edges_url = url_for(
            "main.view_classical_result",
            media_id=media.id,
            result_type="edges",
        )

        sift_image_url = url_for(
            "main.view_feature_result",
            media_id=media.id,
            algorithm="sift",
        )

        hog_image_url = url_for(
            "main.view_feature_result",
            media_id=media.id,
            algorithm="hog",
        )

        return render_template(
            "comparison.html",
            media=media,
            yolo_analysis=yolo_analysis,
            yolo_time=yolo_time,
            classical_analysis=classical_analysis,
            sift_analysis=sift_analysis,
            hog_analysis=hog_analysis,
            yolo_image_url=yolo_image_url,
            grayscale_url=grayscale_url,
            edges_url=edges_url,
            sift_image_url=sift_image_url,
            hog_image_url=hog_image_url,
        )

    except (OSError, ValueError, RuntimeError) as error:
        current_app.logger.exception(
            "Image comparison failed for media ID %s.",
            media.id,
        )

        flash(
            f"Could not compare this image: {error}",
            "error",
        )

        return redirect(url_for("main.dashboard"))

    except Exception:
        current_app.logger.exception(
            "Unexpected image comparison error for media ID %s.",
            media.id,
        )

        flash(
            "An unexpected error occurred during comparison. "
            "Check the application terminal for details.",
            "error",
        )

        return redirect(url_for("main.dashboard"))


@main.route("/feature-result/<int:media_id>/<algorithm>")
@login_required
def view_feature_result(media_id, algorithm):
    """Serve the SIFT or HOG result for owned images."""

    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    if media.file_type != "image":
        abort(404)

    if algorithm not in {"sift", "hog"}:
        abort(404)

    _, stored_path = get_safe_media_path(media)
    image_stem = Path(stored_path).stem
    output_filename = f"{image_stem}_{algorithm}_features.jpg"

    project_dir = Path(current_app.root_path).parent
    results_directory = project_dir / "results" / algorithm
    output_path = results_directory / output_filename

    if not output_path.is_file():
        abort(404)

    response = send_from_directory(
        str(results_directory),
        output_filename,
        mimetype="image/jpeg",
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@main.route("/classical-result/<int:media_id>/<result_type>")
@login_required
def view_classical_result(media_id, result_type):
    """Serve a generated grayscale or edge-detection image."""

    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    if media.file_type != "image":
        abort(404)

    if result_type not in {"grayscale", "edges"}:
        abort(404)

    _, stored_path = get_safe_media_path(media)
    result_filename = f"{Path(stored_path).stem}_{result_type}.jpg"

    results_directory = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "classical"
    )

    result_path = results_directory / result_filename

    if not result_path.is_file():
        abort(404)

    response = send_from_directory(
        str(results_directory),
        result_filename,
        mimetype="image/jpeg",
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@main.route("/analyze/<int:media_id>")
@login_required
def analyze_media(media_id):
    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    if media.file_type != "image":
        flash(
            "Image analysis is currently available for images only.",
            "error",
        )
        return redirect(url_for("main.dashboard"))

    _, stored_path = get_safe_media_path(media)

    try:
        result = analyze_image(stored_path)

        analysis = {
            "filename": result["filename"],
            "width": result["width"],
            "height": result["height"],
            "channels": result["channels"],
            "file_size_kb": result["file_size_kb"],
            "grayscale_status": "Successful",
            "detections": result["detections"],
            "detection_count": result["detection_count"],
        }

        annotated_image_url = url_for(
            "main.view_analysis_result",
            media_id=media.id,
        )

    except (FileNotFoundError, ValueError, OSError):
        current_app.logger.exception(
            "Image analysis failed for media ID %s.",
            media.id,
        )

        flash(
            "This image could not be processed. Please check the uploaded file.",
            "error",
        )

        return redirect(url_for("main.dashboard"))

    except Exception:
        current_app.logger.exception(
            "Unexpected image-processing error for media ID %s.",
            media.id,
        )

        flash(
            "An unexpected error occurred during image analysis. "
            "Check the application terminal for details.",
            "error",
        )

        return redirect(url_for("main.dashboard"))

    return render_template(
        "analyze.html",
        media=media,
        analysis=analysis,
        annotated_image_url=annotated_image_url,
    )


@main.route("/analysis-result/<int:media_id>")
@login_required
def view_analysis_result(media_id):
    """Serve an annotated image belonging to the current user."""

    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    if media.file_type != "image":
        abort(404)

    results_folder, output_filename = get_detection_result_path(media)

    response = send_from_directory(
        results_folder,
        output_filename,
        mimetype="image/jpeg",
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@main.route("/analyze-video/<int:media_id>")
@login_required
def analyze_video_media(media_id):
    """Process an uploaded video using the configured wildlife detector."""

    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    if media.file_type != "video":
        flash(
            "Video analysis is available for video files only.",
            "error",
        )
        return redirect(url_for("main.dashboard"))

    _, stored_path = get_safe_media_path(media)

    processing_job = ProcessingJob(
        user_id=current_user.id,
        media_id=media.id,
        algorithm="YOLO26n",
        status="processing",
    )

    db.session.add(processing_job)
    db.session.commit()

    processing_job.started_at = db.func.now()
    db.session.commit()

    try:
        result = analyze_video(stored_path)

        processing_job.status = "completed"
        processing_job.completed_at = db.func.now()

        benchmark = AlgorithmBenchmark(
            processing_job_id=processing_job.id,
            algorithm="YOLO26n",
            processing_time=result["processing_time_seconds"],
            fps=result["processing_fps"],
        )

        db.session.add(benchmark)
        db.session.commit()

        return render_template(
            "video_analysis.html",
            media=media,
            analysis=result,
            video_url=url_for(
                "main.view_video_result",
                media_id=media.id,
            ),
        )

    except (FileNotFoundError, ValueError, OSError, RuntimeError):
        db.session.rollback()

        current_app.logger.exception(
            "Video analysis failed for media ID %s.",
            media.id,
        )

        try:
            processing_job.status = "failed"
            processing_job.completed_at = db.func.now()
            db.session.commit()
        except Exception:
            db.session.rollback()

        flash(
            "This video could not be processed. Please check the uploaded file.",
            "error",
        )

        return redirect(url_for("main.dashboard"))

    except Exception:
        db.session.rollback()

        current_app.logger.exception(
            "Unexpected video-processing error for media ID %s.",
            media.id,
        )

        try:
            processing_job.status = "failed"
            processing_job.completed_at = db.func.now()
            db.session.commit()
        except Exception:
            db.session.rollback()

        flash(
            "An unexpected error occurred during video analysis. "
            "Check the application terminal for details.",
            "error",
        )

        return redirect(url_for("main.dashboard"))


@main.route("/compare-video/<int:media_id>")
@login_required
def compare_video(media_id):
    """Compare YOLO, OpenCV edges, SIFT features, and HOG features on video."""

    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    if media.file_type != "video":
        flash(
            "Video comparison is available for video files only.",
            "error",
        )
        return redirect(url_for("main.dashboard"))

    _, stored_path = get_safe_media_path(media)

    project_dir = Path(current_app.root_path).parent
    video_results_dir = project_dir / "results" / "videos"
    video_results_dir.mkdir(parents=True, exist_ok=True)

    yolo_filename = f"{Path(stored_path).stem}_detected.mp4"
    yolo_path = video_results_dir / yolo_filename

    try:
        # Reuse an existing YOLO result and its latest saved benchmark
        # when both are available.
        yolo_analysis = None

        if yolo_path.is_file():
            previous_job = (
                ProcessingJob.query
                .filter_by(
                    user_id=current_user.id,
                    media_id=media.id,
                    algorithm="YOLO26n",
                    status="completed",
                )
                .order_by(ProcessingJob.id.desc())
                .first()
            )

            if previous_job is not None:
                previous_benchmark = AlgorithmBenchmark.query.filter_by(
                    processing_job_id=previous_job.id
                ).first()

                if previous_benchmark is not None:
                    yolo_analysis = {
                        "processing_time_seconds":
                            previous_benchmark.processing_time,
                        "processing_fps": previous_benchmark.fps,
                    }

        # Generate YOLO output if a reusable result and benchmark
        # are not available. This does not create a duplicate job.
        if yolo_analysis is None:
            yolo_analysis = analyze_video(stored_path)

        # Generate OpenCV edge, SIFT keypoint, and HOG gradient videos.
        feature_analysis = analyze_video_features(stored_path)

        return render_template(
            "video_comparison.html",
            media=media,
            yolo_analysis=yolo_analysis,
            feature_analysis=feature_analysis,
            yolo_video_url=url_for(
                "main.view_video_result",
                media_id=media.id,
            ),
            edges_video_url=url_for(
                "main.view_video_comparison_result",
                media_id=media.id,
                algorithm="edges",
            ),
            sift_video_url=url_for(
                "main.view_video_comparison_result",
                media_id=media.id,
                algorithm="sift",
            ),
            hog_video_url=url_for(
                "main.view_video_comparison_result",
                media_id=media.id,
                algorithm="hog",
            ),
        )

    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        current_app.logger.exception(
            "Video comparison failed for media ID %s.",
            media.id,
        )

        flash(
            f"Could not compare this video: {error}",
            "error",
        )

        return redirect(url_for("main.dashboard"))

    except Exception:
        current_app.logger.exception(
            "Unexpected video comparison error for media ID %s.",
            media.id,
        )

        flash(
            "An unexpected error occurred during video comparison. "
            "Check the application terminal for details.",
            "error",
        )

        return redirect(url_for("main.dashboard"))


@main.route(
    "/video-comparison-result/<int:media_id>/<algorithm>"
)
@login_required
def view_video_comparison_result(media_id, algorithm):
    """Serve generated feature videos for the media owner."""

    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    if media.file_type != "video":
        abort(404)

    if algorithm not in {"edges", "sift", "hog"}:
        abort(404)

    _, stored_path = get_safe_media_path(media)

    media_stem = Path(stored_path).stem

    output_filenames = {
        "edges": f"{media_stem}_edges.mp4",
        "sift": f"{media_stem}_sift_features.mp4",
        "hog": f"{media_stem}_hog_features.mp4",
    }

    output_filename = output_filenames[algorithm]

    results_directory = (
        Path(current_app.root_path).parent
        / "results"
        / "videos"
    )

    output_path = results_directory / output_filename

    if not output_path.is_file():
        abort(404)

    response = send_from_directory(
        str(results_directory),
        output_filename,
        mimetype="video/mp4",
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@main.route("/video-analysis-result/<int:media_id>")
@login_required
def view_video_result(media_id):
    """Serve a processed video belonging to the current user."""

    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    if media.file_type != "video":
        abort(404)

    results_folder, output_filename = get_video_result_path(media)

    response = send_from_directory(
        results_folder,
        output_filename,
        mimetype="video/mp4",
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@main.route("/logout")
@login_required
def logout():
    logout_user()

    flash(
        "You have logged out successfully.",
        "success",
    )

    return redirect(url_for("main.login"))


@main.errorhandler(413)
def upload_too_large(error):
    flash(
        "The upload exceeds the 100 MB request limit.",
        "error",
    )

    return redirect(url_for("main.upload_media"))
