
import io
import json
import os
import time
import uuid
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

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
    send_file,
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
    upload_folder = os.path.realpath(current_app.config["UPLOAD_FOLDER"])
    stored_path = os.path.realpath(media.file_path)

    try:
        if os.path.commonpath([upload_folder, stored_path]) != upload_folder:
            abort(404)
    except ValueError:
        abort(404)

    if not os.path.isfile(stored_path):
        abort(404)

    return upload_folder, stored_path


def get_result_file(folder_name, filename):
    project_dir = Path(current_app.root_path).parent

    results_folder = os.path.realpath(project_dir / "results" / folder_name)
    output_path = os.path.realpath(os.path.join(results_folder, filename))

    try:
        if os.path.commonpath([results_folder, output_path]) != results_folder:
            abort(404)
    except ValueError:
        abort(404)

    if not os.path.isfile(output_path):
        abort(404)

    return results_folder, os.path.basename(output_path)


def optional_result_file(folder_name, filename):
    """Return an existing result file, or None if it does not exist."""
    project_dir = Path(current_app.root_path).parent
    folder = os.path.realpath(project_dir / "results" / folder_name)
    path = os.path.realpath(os.path.join(folder, filename))

    try:
        if os.path.commonpath([folder, path]) != folder:
            return None
    except ValueError:
        return None

    if not os.path.isfile(path):
        return None

    return folder, os.path.basename(path)


def get_detection_result_path(media):
    _, stored_path = get_safe_media_path(media)
    return get_result_file(
        "detections",
        Path(stored_path).stem + "_detected.jpg",
    )


def get_video_result_path(media):
    _, stored_path = get_safe_media_path(media)
    return get_result_file(
        "videos",
        Path(stored_path).stem + "_detected.mp4",
    )


def get_safe_deletion_paths(media):
    upload_folder = os.path.realpath(current_app.config["UPLOAD_FOLDER"])
    stored_path = os.path.realpath(media.file_path)

    try:
        if os.path.commonpath([upload_folder, stored_path]) != upload_folder:
            abort(404)
    except ValueError:
        abort(404)

    project_dir = Path(current_app.root_path).parent
    stem = Path(stored_path).stem
    paths = [stored_path]

    definitions = {
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

    for folder_name, filenames in definitions.items():
        folder = os.path.realpath(project_dir / "results" / folder_name)

        for filename in filenames:
            path = os.path.realpath(os.path.join(folder, filename))

            try:
                if os.path.commonpath([folder, path]) != folder:
                    abort(404)
            except ValueError:
                abort(404)

            paths.append(path)

    return paths


def get_owned_media(media_id):
    media = db.session.get(Media, media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    return media


def get_owned_job(job_id):
    job = db.session.get(ProcessingJob, job_id)

    if job is None:
        abort(404)

    if job.user_id != current_user.id:
        abort(403)

    return job


# ---------------------------------------------------------
# SAVE RESULTS AS JSON
# ---------------------------------------------------------

def json_safe(value):
    """Convert common processor results into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, Path):
        return str(value)

    return str(value)


def create_processing_job(media, algorithm, comparison_id=None):
    job = ProcessingJob(
        user_id=current_user.id,
        media_id=media.id,
        algorithm=algorithm,
        comparison_id=comparison_id,
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
        processing_time = result.get("processing_time_seconds", elapsed)
        fps = result.get("processing_fps")

    job.status = "completed"
    job.completed_at = db.func.now()
    job.result_data = json.dumps(json_safe(result or {}))

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


def build_output_links(job, media):
    """Build links only for output files that currently exist."""
    _, stored_path = get_safe_media_path(media)
    stem = Path(stored_path).stem
    links = []

    def add_file(folder, filename, label, endpoint, **kwargs):
        if optional_result_file(folder, filename):
            links.append({
                "label": label,
                "url": url_for(endpoint, **kwargs),
                "download": False,
            })

    if media.file_type == "image":
        if job.algorithm == "YOLO26n":
            add_file(
                "detections",
                f"{stem}_detected.jpg",
                "View detected image",
                "main.view_analysis_result",
                media_id=media.id,
            )

        elif job.algorithm == "SIFT":
            add_file(
                "sift",
                f"{stem}_sift_features.jpg",
                "View SIFT output",
                "main.view_feature_result",
                media_id=media.id,
                algorithm="sift",
            )

        elif job.algorithm == "HOG":
            add_file(
                "hog",
                f"{stem}_hog_features.jpg",
                "View HOG output",
                "main.view_feature_result",
                media_id=media.id,
                algorithm="hog",
            )

        elif job.algorithm == "Classical OpenCV":
            for result_type in ("grayscale", "edges"):
                add_file(
                    "classical",
                    f"{stem}_{result_type}.jpg",
                    f"View {result_type} image",
                    "main.view_classical_result",
                    media_id=media.id,
                    result_type=result_type,
                )

    elif job.algorithm == "YOLO26n":
        add_file(
            "videos",
            f"{stem}_detected.mp4",
            "View processed video",
            "main.view_video_result",
            media_id=media.id,
        )

    else:
        for algorithm in ("edges", "sift", "hog"):
            filename = (
                f"{stem}_edges.mp4"
                if algorithm == "edges"
                else f"{stem}_{algorithm}_features.mp4"
            )

            add_file(
                "videos",
                filename,
                f"View {algorithm} video",
                "main.view_video_comparison_result",
                media_id=media.id,
                algorithm=algorithm,
            )

    return links


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

        flash("Account created successfully. You can now log in.", "success")
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
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for("main.dashboard"))

        flash("Invalid email or password. Please try again.", "error")
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

    processing_jobs = ProcessingJob.query.filter_by(
        user_id=current_user.id
    ).order_by(ProcessingJob.started_at.desc()).all()

    comparison_jobs = ProcessingJob.query.filter(
        ProcessingJob.user_id == current_user.id,
        ProcessingJob.comparison_id.isnot(None),
    ).order_by(ProcessingJob.started_at.desc()).all()

    seen_comparisons = set()
    comparison_history = []

    for job in comparison_jobs:
        if job.comparison_id in seen_comparisons:
            continue

        seen_comparisons.add(job.comparison_id)

        grouped_jobs = [
            item for item in comparison_jobs
            if item.comparison_id == job.comparison_id
        ]

        comparison_history.append({
            "comparison_id": job.comparison_id,
            "media": job.media,
            "started_at": job.started_at,
            "algorithms": [item.algorithm for item in grouped_jobs],
            "status": (
                "completed"
                if grouped_jobs and all(
                    item.status == "completed"
                    for item in grouped_jobs
                )
                else "incomplete"
            ),
        })

    return render_template(
        "dashboard.html",
        media_files=media_files,
        processing_jobs=processing_jobs,
        comparison_history=comparison_history,
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

        flash(f"{original_filename} uploaded successfully.", "success")
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
        jobs = ProcessingJob.query.filter_by(media_id=media.id).all()
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
            "Could not delete media record %s.", media_id
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
                "Could not remove stored file: %s", path
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
# SAVED INDIVIDUAL RESULTS AND DOWNLOADS
# ---------------------------------------------------------

@main.route("/results/<int:job_id>")
@login_required
def saved_result(job_id):
    job = get_owned_job(job_id)

    if job.status != "completed":
        flash(
            "This processing job has no completed results to display.",
            "error",
        )
        return redirect(url_for("main.dashboard"))

    media = db.session.get(Media, job.media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    try:
        result = json.loads(job.result_data or "{}")
    except (TypeError, json.JSONDecodeError):
        result = {}

    if not isinstance(result, dict):
        result = {"result": result}

    result_items = [
        {
            "label": str(key).replace("_", " ").title(),
            "value": value,
        }
        for key, value in result.items()
    ]

    benchmark = AlgorithmBenchmark.query.filter_by(
        processing_job_id=job.id
    ).first()

    metrics = []

    if benchmark:
        if benchmark.processing_time is not None:
            metrics.append({
                "label": "Processing time (seconds)",
                "value": benchmark.processing_time,
            })

        if benchmark.fps is not None:
            metrics.append({
                "label": "Processing FPS",
                "value": benchmark.fps,
            })

    output_links = build_output_links(job, media)

    return render_template(
        "results.html",
        job=job,
        media=media,
        result=result,
        result_items=result_items,
        metrics=metrics,
        output_links=output_links,
    )


@main.route("/download-result/<int:job_id>")
@login_required
def download_result(job_id):
    job = get_owned_job(job_id)

    if job.status != "completed":
        abort(404)

    media = db.session.get(Media, job.media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    _, stored_path = get_safe_media_path(media)
    stem = Path(stored_path).stem
    selected = None
    download_name = None

    if media.file_type == "image":
        if job.algorithm == "YOLO26n":
            selected = optional_result_file(
                "detections", f"{stem}_detected.jpg"
            )
            download_name = f"{stem}_detected.jpg"

        elif job.algorithm == "SIFT":
            selected = optional_result_file(
                "sift", f"{stem}_sift_features.jpg"
            )
            download_name = f"{stem}_sift_features.jpg"

        elif job.algorithm == "HOG":
            selected = optional_result_file(
                "hog", f"{stem}_hog_features.jpg"
            )
            download_name = f"{stem}_hog_features.jpg"

        elif job.algorithm == "Classical OpenCV":
            selected = optional_result_file(
                "classical", f"{stem}_edges.jpg"
            )
            download_name = f"{stem}_edges.jpg"

    elif job.algorithm == "YOLO26n":
        selected = optional_result_file(
            "videos", f"{stem}_detected.mp4"
        )
        download_name = f"{stem}_detected.mp4"

    else:
        candidates = [
            ("videos", f"{stem}_edges.mp4"),
            ("videos", f"{stem}_sift_features.mp4"),
            ("videos", f"{stem}_hog_features.mp4"),
        ]

        existing = [
            (folder, filename, optional_result_file(folder, filename))
            for folder, filename in candidates
        ]
        existing = [item for item in existing if item[2] is not None]

        if not existing:
            abort(404)

        memory_file = io.BytesIO()

        with zipfile.ZipFile(
            memory_file,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for _, filename, (folder, safe_name) in existing:
                archive.write(
                    os.path.join(folder, safe_name),
                    arcname=safe_name,
                )

        memory_file.seek(0)

        return send_file(
            memory_file,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{stem}_classical_video_results.zip",
        )

    if selected is None:
        flash(
            "The processed output file is not available.",
            "error",
        )
        return redirect(url_for("main.saved_result", job_id=job.id))

    folder, filename = selected

    return send_from_directory(
        folder,
        filename,
        as_attachment=True,
        download_name=download_name,
        conditional=True,
    )


@main.route("/download-report/<int:job_id>")
@login_required
def download_report(job_id):
    job = get_owned_job(job_id)

    if job.status != "completed":
        abort(404)

    media = db.session.get(Media, job.media_id)

    if media is None:
        abort(404)

    if media.user_id != current_user.id:
        abort(403)

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Preformatted,
        )
    except ImportError:
        flash(
            "PDF reporting requires ReportLab. Install it using "
            "'python -m pip install \"reportlab>=4.2,<5\"'.",
            "error",
        )
        return redirect(url_for("main.saved_result", job_id=job.id))

    try:
        result = json.loads(job.result_data or "{}")
    except (TypeError, json.JSONDecodeError):
        result = {}

    benchmark = AlgorithmBenchmark.query.filter_by(
        processing_job_id=job.id
    ).first()

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"Wildlife Comparative Analyzer - Job {job.id}",
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph("Wildlife Comparative Analyzer", styles["Title"]),
        Paragraph("Processing Results Report", styles["Heading2"]),
        Spacer(1, 8 * mm),
    ]

    details = [
        ["Property", "Value"],
        ["Processing Job ID", str(job.id)],
        ["Media File", str(media.filename)],
        ["Media Type", str(media.file_type)],
        ["Algorithm", str(job.algorithm)],
        ["Status", str(job.status)],
        ["Started", str(job.started_at or "Not recorded")],
        ["Completed", str(job.completed_at or "Not recorded")],
    ]

    if benchmark:
        if benchmark.processing_time is not None:
            details.append([
                "Processing Time (seconds)",
                str(benchmark.processing_time),
            ])

        if benchmark.fps is not None:
            details.append(["Processing FPS", str(benchmark.fps)])

    table = Table(
        details,
        colWidths=[55 * mm, 105 * mm],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263746")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.extend([table, Spacer(1, 8 * mm)])
    story.append(Paragraph("Analysis Details", styles["Heading2"]))

    if isinstance(result, dict) and result:
        for key, value in result.items():
            label = escape(str(key).replace("_", " ").title())
            serialized = json.dumps(
                json_safe(value),
                indent=2,
                ensure_ascii=False,
            )

            if len(serialized) > 5000:
                serialized = (
                    serialized[:5000]
                    + "\n... (truncated in PDF report)"
                )

            story.append(Paragraph(label, styles["Heading4"]))
            story.append(Preformatted(
                escape(serialized),
                styles["Code"],
                maxLineLength=95,
            ))
            story.append(Spacer(1, 3 * mm))
    else:
        story.append(Paragraph(
            "No saved analysis details were available.",
            styles["BodyText"],
        ))

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "This report includes only metrics recorded by the application. "
        "Detection counts or feature measurements should not be interpreted "
        "as accuracy, precision, recall, F1-score, or mAP unless those metrics "
        "were explicitly calculated and recorded.",
        styles["Italic"],
    ))

    document.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"wildlife_analysis_job_{job.id}.pdf",
    )


# ---------------------------------------------------------
# ANALYZE USING THE USER'S SELECTED ALGORITHM
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
        started = time.perf_counter()

        if media.file_type == "image":
            if algorithm == "YOLO26n":
                result = analyze_image(stored_path)
            elif algorithm == "Classical OpenCV":
                result = analyze_classical_image(stored_path)
            elif algorithm == "SIFT":
                result = analyze_sift_image(stored_path)
            else:
                result = analyze_hog_image(stored_path)

        elif algorithm == "YOLO26n":
            result = analyze_video(stored_path)
        else:
            result = analyze_video_features(stored_path)

        elapsed = round(time.perf_counter() - started, 4)
        complete_processing_job(job, algorithm, elapsed, result)

        return redirect(
            url_for("main.saved_result", job_id=job.id)
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
# IMAGE COMPARISON: RUN ALL FOUR METHODS AND SAVE THE GROUP
# ---------------------------------------------------------

@main.route("/compare/<int:media_id>")
@login_required
def compare_image(media_id):
    media = get_owned_media(media_id)

    if media.file_type != "image":
        flash("Image comparison is available for images only.", "error")
        return redirect(url_for("main.dashboard"))

    _, stored_path = get_safe_media_path(media)
    comparison_id = str(uuid.uuid4())

    def run_benchmark(algorithm, processor):
        job = create_processing_job(
            media,
            algorithm,
            comparison_id=comparison_id,
        )

        try:
            started = time.perf_counter()
            result = processor(stored_path)
            elapsed = round(time.perf_counter() - started, 4)
            complete_processing_job(job, algorithm, elapsed, result)
            return result, elapsed

        except Exception:
            fail_processing_job(job.id)
            raise

    try:
        run_benchmark("YOLO26n", analyze_image)
        run_benchmark("Classical OpenCV", analyze_classical_image)
        run_benchmark("SIFT", analyze_sift_image)
        run_benchmark("HOG", analyze_hog_image)

        return redirect(
            url_for(
                "main.saved_comparison",
                comparison_id=comparison_id,
            )
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
# SAVED COMPARISON RESULTS
# ---------------------------------------------------------

@main.route("/comparison-results/<comparison_id>")
@login_required
def saved_comparison(comparison_id):
    jobs = ProcessingJob.query.filter_by(
        comparison_id=comparison_id,
        user_id=current_user.id,
    ).order_by(ProcessingJob.id.asc()).all()

    if not jobs:
        abort(404)

    media_ids = {job.media_id for job in jobs}

    if len(media_ids) != 1:
        current_app.logger.error(
            "Comparison %s contains jobs for multiple media files.",
            comparison_id,
        )
        abort(400)

    media = db.session.get(Media, jobs[0].media_id)

    if media is None or media.user_id != current_user.id:
        abort(404)

    comparisons = []

    for job in jobs:
        try:
            result = json.loads(job.result_data or "{}")
        except (TypeError, json.JSONDecodeError):
            result = {}

        if not isinstance(result, dict):
            result = {"result": result}

        benchmark = AlgorithmBenchmark.query.filter_by(
            processing_job_id=job.id
        ).first()

        comparisons.append({
            "job": job,
            "result": result,
            "processing_time": (
                benchmark.processing_time if benchmark else None
            ),
            "fps": benchmark.fps if benchmark else None,
            "output_links": (
                build_output_links(job, media)
                if job.status == "completed"
                else []
            ),
        })

    return render_template(
        "comparison_results.html",
        comparison_id=comparison_id,
        media=media,
        comparisons=comparisons,
    )


# ---------------------------------------------------------
# SERVE IMAGE RESULTS
# ---------------------------------------------------------

@main.route("/feature-result/<int:media_id>/<algorithm>")
@login_required
def view_feature_result(media_id, algorithm):
    media = get_owned_media(media_id)

    if media.file_type != "image" or algorithm not in {"sift", "hog"}:
        abort(404)

    _, stored_path = get_safe_media_path(media)
    filename = f"{Path(stored_path).stem}_{algorithm}_features.jpg"
    folder, filename = get_result_file(algorithm, filename)

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

    if media.file_type != "image" or result_type not in {
        "grayscale", "edges"
    }:
        abort(404)

    _, stored_path = get_safe_media_path(media)
    filename = f"{Path(stored_path).stem}_{result_type}.jpg"
    folder, filename = get_result_file("classical", filename)

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
        flash("Video analysis is available for video files only.", "error")
        return redirect(url_for("main.dashboard"))

    _, stored_path = get_safe_media_path(media)
    job = create_processing_job(media, "YOLO26n")

    try:
        started = time.perf_counter()
        result = analyze_video(stored_path)
        elapsed = round(time.perf_counter() - started, 4)
        complete_processing_job(job, "YOLO26n", elapsed, result)

        return redirect(
            url_for("main.saved_result", job_id=job.id)
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
# VIDEO COMPARISON: RUN BOTH METHODS AND SAVE THE GROUP
# ---------------------------------------------------------

@main.route("/compare-video/<int:media_id>")
@login_required
def compare_video(media_id):
    media = get_owned_media(media_id)

    if media.file_type != "video":
        flash("Video comparison is available for video files only.", "error")
        return redirect(url_for("main.dashboard"))

    _, stored_path = get_safe_media_path(media)
    comparison_id = str(uuid.uuid4())

    def run_video_benchmark(algorithm, processor):
        job = create_processing_job(
            media,
            algorithm,
            comparison_id=comparison_id,
        )

        try:
            started = time.perf_counter()
            result = processor(stored_path)
            elapsed = round(time.perf_counter() - started, 4)
            complete_processing_job(job, algorithm, elapsed, result)
            return result

        except Exception:
            fail_processing_job(job.id)
            raise

    try:
        run_video_benchmark("YOLO26n", analyze_video)
        run_video_benchmark(
            "Classical Video Features (Canny + SIFT + HOG)",
            analyze_video_features,
        )

        return redirect(
            url_for(
                "main.saved_comparison",
                comparison_id=comparison_id,
            )
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


@main.route("/video-comparison-result/<int:media_id>/<algorithm>")
@login_required
def view_video_comparison_result(media_id, algorithm):
    media = get_owned_media(media_id)

    if media.file_type != "video" or algorithm not in {
        "edges", "sift", "hog"
    }:
        abort(404)

    _, stored_path = get_safe_media_path(media)
    stem = Path(stored_path).stem

    filename = {
        "edges": f"{stem}_edges.mp4",
        "sift": f"{stem}_sift_features.mp4",
        "hog": f"{stem}_hog_features.mp4",
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
    flash("You have logged out successfully.", "success")
    return redirect(url_for("main.login"))


@main.errorhandler(413)
def upload_too_large(error):
    flash("The upload exceeds the 100 MB request limit.", "error")
    return redirect(url_for("main.upload_media"))
