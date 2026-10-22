
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

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from app import db
from app.models import User, Media, ProcessingJob, AlgorithmBenchmark

from algorithms.classical_processor import analyze_classical_image
from algorithms.image_processor import analyze_image
from algorithms.video_processor import analyze_video
from algorithms.video_feature_processor import analyze_video_features
from algorithms.sift_processor import analyze_sift_image
from algorithms.hog_processor import analyze_hog_image


main = Blueprint("main", __name__)

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


# ---------------------------------------------------------
# FILE AND PATH HELPERS
# ---------------------------------------------------------

def get_file_type(filename):
    if "." not in filename:
        return None

    extension = filename.rsplit(".", 1)[1].lower()

    if extension in ALLOWED_IMAGE_EXTENSIONS:
        return "image"

    if extension in ALLOWED_VIDEO_EXTENSIONS:
        return "video"

    return None


def get_safe_media_path(media):
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


def get_result_file(folder_name, filename):
    project_dir = Path(current_app.root_path).parent

    results_folder = os.path.realpath(
        project_dir / "results" / folder_name
    )
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

    if not os.path.isfile(output_path):
        abort(404)

    return results_folder, os.path.basename(output_path)


def get_detection_result_path(media):
    _, stored_path = get_safe_media_path(media)

    filename = Path(stored_path).stem + "_detected.jpg"

    return get_result_file("detections", filename)


def get_video_result_path(media):
    _, stored_path = get_safe_media_path(media)

    filename = Path(stored_path).stem + "_detected.mp4"

    return get_result_file("videos", filename)


def get_safe_deletion_paths(media):
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
    stem = Path(stored_path).stem

    paths = [stored_path]

    result_definitions = {
        "detections": [f"{stem}_detected.jpg"],
        "videos": [
            f"{stem}_detected.mp4",
            f"{stem}_edges.mp4",
            f"{stem}_sift_features.mp4",
            f"{stem}_hog_features.mp4",
        ],
        "sift": [f"{stem}_sift_features.jpg"],
        "hog": [f"{stem}_hog_features.jpg"],
        "classical": [
            f"{stem}_grayscale.jpg",
            f"{stem}_edges.jpg",
        ],
    }

    for folder_name, filenames in result_definitions.items():
        folder = os.path.realpath(
            project_dir / "results" / folder_name
        )

        for filename in filenames:
            output_path = os.path.realpath(
                os.path.join(folder, filename)
            )

            try:
                if os.path.commonpath(
                    [folder, output_path]
                ) != folder:
                    abort(404)
            except ValueError:
                abort(404)

            paths.append(output_path)

    return paths


def get_owned_media(media_id):
    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    return media


# ---------------------------------------------------------
# PROCESSING JOB AND BENCHMARK HELPERS
# ---------------------------------------------------------

def create_processing_job(media, algorithm):
    job = ProcessingJob(
        user_id=current_user.id,
        media_id=media.id,
        algorithm=algorithm,
        status="processing",
        started_at=db.func.now(),
    )

    db.session.add(job)
    db.session.commit()

    return job


def complete_processing_job(job, algorithm, elapsed, result=None):
    processing_time = elapsed
    fps = None

    if isinstance(result, dict):
        processing_time = result.get(
            "processing_time_seconds", elapsed
        )
        fps = result.get("processing_fps")

    job.status = "completed"
    job.completed_at = db.func.now()

    benchmark = AlgorithmBenchmark(
        processing_job_id=job.id,
        algorithm=algorithm,
        processing_time=processing_time,
        fps=fps,
    )

    db.session.add(benchmark)
    db.session.commit()


def fail_processing_job(job_id):
    db.session.rollback()

    try:
        job = db.session.get(ProcessingJob, job_id)

        if job is not None:
            job.status = "failed"
            job.completed_at = db.func.now()
            db.session.commit()

    except Exception:
        db.session.rollback()
        current_app.logger.exception(
            "Could not mark processing job as failed."
        )


# ---------------------------------------------------------
# HOME, REGISTRATION AND LOGIN
# ---------------------------------------------------------

@main.route("/")
def home():
    return render_template("index.html")


@main.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

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
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

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


# ---------------------------------------------------------
# DASHBOARD AND UPLOAD
# ---------------------------------------------------------

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
                "Unsupported file type. Select a supported image or video.",
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
                        "Could not remove failed upload."
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


# ---------------------------------------------------------
# VIEW AND DELETE UPLOADED MEDIA
# ---------------------------------------------------------

@main.route("/media/<int:media_id>")
@login_required
def view_media(media_id):
    media = get_owned_media(media_id)
    folder, stored_path = get_safe_media_path(media)

    response = send_from_directory(
        folder,
        os.path.basename(stored_path),
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@main.route("/delete-media/<int:media_id>", methods=["POST"])
@login_required
def delete_media(media_id):
    media = get_owned_media(media_id)
    paths = get_safe_deletion_paths(media)

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

    cleanup_failed = False

    for path in paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            cleanup_failed = True
            current_app.logger.exception(
                "Could not remove stored file: %s",
                path,
            )

    if cleanup_failed:
        flash(
            "The media record was deleted, but one or more stored files "
            "could not be removed.",
            "error",
        )
    else:
        flash("Media deleted successfully.", "success")

    return redirect(url_for("main.dashboard"))


# ---------------------------------------------------------
# NEW: ANALYZE USING THE USER'S SELECTED ALGORITHM
# ---------------------------------------------------------

@main.route("/analyze-selected/<int:media_id>", methods=["POST"])
@login_required
def analyze_selected(media_id):
    media = get_owned_media(media_id)
    algorithm = request.form.get("algorithm", "").strip()

    image_algorithms = {
        "YOLO26n",
        "Classical OpenCV",
        "SIFT",
        "HOG",
    }

    video_algorithms = {
        "YOLO26n",
        "Classical Video Features",
    }

    if media.file_type == "image" and algorithm not in image_algorithms:
        flash("Choose a valid algorithm for this image.", "error")
        return redirect(url_for("main.dashboard"))

    if media.file_type == "video" and algorithm not in video_algorithms:
        flash("Choose a valid algorithm for this video.", "error")
        return redirect(url_for("main.dashboard"))

    job_id = None

    try:
        _, stored_path = get_safe_media_path(media)

        job = create_processing_job(media, algorithm)
        job_id = job.id

        start_time = time.perf_counter()

        # IMAGE ANALYSIS
        if media.file_type == "image":

            if algorithm == "YOLO26n":
                result = analyze_image(stored_path)

                elapsed = round(
                    time.perf_counter() - start_time, 4
                )

                complete_processing_job(
                    job, algorithm, elapsed, result
                )

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

                return render_template(
                    "analyze.html",
                    media=media,
                    analysis=analysis,
                    annotated_image_url=url_for(
                        "main.view_analysis_result",
                        media_id=media.id,
                    ),
                )

            if algorithm == "Classical OpenCV":
                result = analyze_classical_image(stored_path)
                result_type = "classical"
                result_url = None

            elif algorithm == "SIFT":
                result = analyze_sift_image(stored_path)
                result_type = "sift"

                result_url = url_for(
                    "main.view_feature_result",
                    media_id=media.id,
                    algorithm="sift",
                )

            else:
                result = analyze_hog_image(stored_path)
                result_type = "hog"

                result_url = url_for(
                    "main.view_feature_result",
                    media_id=media.id,
                    algorithm="hog",
                )

            elapsed = round(
                time.perf_counter() - start_time, 4
            )

            complete_processing_job(
                job, algorithm, elapsed, result
            )

            return render_template(
                "selected_image_analysis.html",
                media=media,
                algorithm=algorithm,
                analysis=result,
                result_type=result_type,
                result_url=result_url,
                grayscale_url=url_for(
                    "main.view_classical_result",
                    media_id=media.id,
                    result_type="grayscale",
                ),
                edges_url=url_for(
                    "main.view_classical_result",
                    media_id=media.id,
                    result_type="edges",
                ),
            )

        # VIDEO ANALYSIS
        if algorithm == "YOLO26n":
            result = analyze_video(stored_path)

            elapsed = round(
                time.perf_counter() - start_time, 4
            )

            complete_processing_job(
                job, algorithm, elapsed, result
            )

            return render_template(
                "video_analysis.html",
                media=media,
                analysis=result,
                video_url=url_for(
                    "main.view_video_result",
                    media_id=media.id,
                ),
            )

        # Classical video feature processing
        result = analyze_video_features(stored_path)

        elapsed = round(
            time.perf_counter() - start_time, 4
        )

        complete_processing_job(
            job, algorithm, elapsed, result
        )

        return render_template(
            "selected_video_analysis.html",
            media=media,
            algorithm=algorithm,
            analysis=result,
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

    except Exception:
        if job_id is not None:
            fail_processing_job(job_id)

        current_app.logger.exception(
            "Selected algorithm %s failed for media ID %s.",
            algorithm,
            media.id,
        )

        flash(
            f"{algorithm or 'The selected algorithm'} could not process "
            "this file. Check the application terminal for details.",
            "error",
        )

        return redirect(url_for("main.dashboard"))


# ---------------------------------------------------------
# IMAGE COMPARISON: RUN ALL FOUR METHODS
# ---------------------------------------------------------

@main.route("/compare/<int:media_id>")
@login_required
def compare_image(media_id):
    media = get_owned_media(media_id)

    if media.file_type != "image":
        flash(
            "Image comparison is available for images only.",
            "error",
        )
        return redirect(url_for("main.dashboard"))

    _, stored_path = get_safe_media_path(media)

    def run_benchmark(algorithm, processor):
        job = create_processing_job(media, algorithm)

        try:
            started = time.perf_counter()
            result = processor(stored_path)

            elapsed = round(
                time.perf_counter() - started, 4
            )

            complete_processing_job(
                job, algorithm, elapsed, result
            )

            return result, elapsed

        except Exception:
            fail_processing_job(job.id)
            raise

    try:
        yolo, yolo_time = run_benchmark(
            "YOLO26n", analyze_image
        )

        classical, classical_time = run_benchmark(
            "Classical OpenCV", analyze_classical_image
        )

        sift, sift_time = run_benchmark(
            "SIFT", analyze_sift_image
        )

        hog, hog_time = run_benchmark(
            "HOG", analyze_hog_image
        )

        return render_template(
            "comparison.html",
            media=media,
            yolo_analysis=yolo,
            yolo_time=yolo_time,
            classical_analysis=classical,
            classical_time=classical_time,
            sift_analysis=sift,
            sift_time=sift_time,
            hog_analysis=hog,
            hog_time=hog_time,
            yolo_image_url=url_for(
                "main.view_analysis_result",
                media_id=media.id,
            ),
            grayscale_url=url_for(
                "main.view_classical_result",
                media_id=media.id,
                result_type="grayscale",
            ),
            edges_url=url_for(
                "main.view_classical_result",
                media_id=media.id,
                result_type="edges",
            ),
            sift_image_url=url_for(
                "main.view_feature_result",
                media_id=media.id,
                algorithm="sift",
            ),
            hog_image_url=url_for(
                "main.view_feature_result",
                media_id=media.id,
                algorithm="hog",
            ),
        )

    except Exception:
        current_app.logger.exception(
            "Image comparison failed for media ID %s.",
            media.id,
        )

        flash(
            "Could not compare this image. Check the application terminal.",
            "error",
        )

        return redirect(url_for("main.dashboard"))


# ---------------------------------------------------------
# SERVE IMAGE RESULTS
# ---------------------------------------------------------

@main.route("/feature-result/<int:media_id>/<algorithm>")
@login_required
def view_feature_result(media_id, algorithm):
    media = get_owned_media(media_id)

    if media.file_type != "image":
        abort(404)

    if algorithm not in {"sift", "hog"}:
        abort(404)

    _, stored_path = get_safe_media_path(media)

    filename = (
        f"{Path(stored_path).stem}_{algorithm}_features.jpg"
    )

    folder, filename = get_result_file(
        algorithm, filename
    )

    response = send_from_directory(
        folder,
        filename,
        mimetype="image/jpeg",
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@main.route("/classical-result/<int:media_id>/<result_type>")
@login_required
def view_classical_result(media_id, result_type):
    media = get_owned_media(media_id)

    if media.file_type != "image":
        abort(404)

    if result_type not in {"grayscale", "edges"}:
        abort(404)

    _, stored_path = get_safe_media_path(media)

    filename = (
        f"{Path(stored_path).stem}_{result_type}.jpg"
    )

    folder, filename = get_result_file(
        "classical", filename
    )

    response = send_from_directory(
        folder,
        filename,
        mimetype="image/jpeg",
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@main.route("/analyze/<int:media_id>")
@login_required
def analyze_media(media_id):
    media = get_owned_media(media_id)

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

        return render_template(
            "analyze.html",
            media=media,
            analysis=analysis,
            annotated_image_url=url_for(
                "main.view_analysis_result",
                media_id=media.id,
            ),
        )

    except Exception:
        current_app.logger.exception(
            "Image analysis failed for media ID %s.",
            media.id,
        )

        flash(
            "This image could not be processed. Check the application terminal.",
            "error",
        )

        return redirect(url_for("main.dashboard"))


@main.route("/analysis-result/<int:media_id>")
@login_required
def view_analysis_result(media_id):
    media = get_owned_media(media_id)

    if media.file_type != "image":
        abort(404)

    folder, filename = get_detection_result_path(media)

    response = send_from_directory(
        folder,
        filename,
        mimetype="image/jpeg",
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


# ---------------------------------------------------------
# VIDEO ANALYSIS
# ---------------------------------------------------------

@main.route("/analyze-video/<int:media_id>")
@login_required
def analyze_video_media(media_id):
    media = get_owned_media(media_id)

    if media.file_type != "video":
        flash(
            "Video analysis is available for video files only.",
            "error",
        )
        return redirect(url_for("main.dashboard"))

    _, stored_path = get_safe_media_path(media)
    job = create_processing_job(media, "YOLO26n")

    try:
        started = time.perf_counter()
        result = analyze_video(stored_path)

        elapsed = round(
            time.perf_counter() - started, 4
        )

        complete_processing_job(
            job, "YOLO26n", elapsed, result
        )

        return render_template(
            "video_analysis.html",
            media=media,
            analysis=result,
            video_url=url_for(
                "main.view_video_result",
                media_id=media.id,
            ),
        )

    except Exception:
        fail_processing_job(job.id)

        current_app.logger.exception(
            "Video analysis failed for media ID %s.",
            media.id,
        )

        flash(
            "This video could not be processed. Check the application terminal.",
            "error",
        )

        return redirect(url_for("main.dashboard"))


# ---------------------------------------------------------
# VIDEO COMPARISON: YOLO AND CLASSICAL VIDEO FEATURES
# ---------------------------------------------------------

@main.route("/compare-video/<int:media_id>")
@login_required
def compare_video(media_id):
    media = get_owned_media(media_id)

    if media.file_type != "video":
        flash(
            "Video comparison is available for video files only.",
            "error",
        )
        return redirect(url_for("main.dashboard"))

    _, stored_path = get_safe_media_path(media)

    def run_video_benchmark(algorithm, processor):
        job = create_processing_job(media, algorithm)

        try:
            started = time.perf_counter()
            result = processor(stored_path)

            elapsed = round(
                time.perf_counter() - started, 4
            )

            complete_processing_job(
                job, algorithm, elapsed, result
            )

            return result

        except Exception:
            fail_processing_job(job.id)
            raise

    try:
        yolo_analysis = run_video_benchmark(
            "YOLO26n", analyze_video
        )

        feature_analysis = run_video_benchmark(
            "Classical Video Features (Canny + SIFT + HOG)",
            analyze_video_features,
        )

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

    except Exception:
        current_app.logger.exception(
            "Video comparison failed for media ID %s.",
            media.id,
        )

        flash(
            "Could not compare this video. Check the application terminal.",
            "error",
        )

        return redirect(url_for("main.dashboard"))


@main.route(
    "/video-comparison-result/<int:media_id>/<algorithm>"
)
@login_required
def view_video_comparison_result(media_id, algorithm):
    media = get_owned_media(media_id)

    if media.file_type != "video":
        abort(404)

    filenames = {
        "edges": f"{Path(media.file_path).stem}_edges.mp4",
        "sift": f"{Path(media.file_path).stem}_sift_features.mp4",
        "hog": f"{Path(media.file_path).stem}_hog_features.mp4",
    }

    if algorithm not in filenames:
        abort(404)

    # Validate the upload path before deriving the output filename.
    _, stored_path = get_safe_media_path(media)
    filename = {
        "edges": f"{Path(stored_path).stem}_edges.mp4",
        "sift": f"{Path(stored_path).stem}_sift_features.mp4",
        "hog": f"{Path(stored_path).stem}_hog_features.mp4",
    }[algorithm]

    folder, filename = get_result_file("videos", filename)

    response = send_from_directory(
        folder,
        filename,
        mimetype="video/mp4",
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


@main.route("/video-analysis-result/<int:media_id>")
@login_required
def view_video_result(media_id):
    media = get_owned_media(media_id)

    if media.file_type != "video":
        abort(404)

    folder, filename = get_video_result_path(media)

    response = send_from_directory(
        folder,
        filename,
        mimetype="video/mp4",
        as_attachment=False,
        conditional=True,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"

    return response


# ---------------------------------------------------------
# LOGOUT AND UPLOAD SIZE ERROR
# ---------------------------------------------------------

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
