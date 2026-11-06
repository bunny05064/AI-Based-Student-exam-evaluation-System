"""Flask application factory."""

import importlib
import re
import uuid

from flask import Flask, render_template, request
from flask_cors import CORS

from app.core.config import BASE_DIR, flask_config, load_settings, public_limits
from app.core.error_handlers import register_error_handlers
from app.core.logging_config import get_logger, request_id_var, setup_logging
from app.extensions import db, migrate

logger = get_logger("app")

# Client-supplied request ids go straight into every log line, so keep them to a
# short, boring character set - otherwise a crafted header can forge log lines.
_REQUEST_ID_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_REQUEST_ID_LEN = 64


_INSECURE_SECRETS = {
    "dev-secret-change-me",
    "change-me-in-production",
    "secret",
    "supersecret",
}


def _guard_production_secrets(settings) -> None:
    """Refuse to serve production traffic with a shipped default SECRET_KEY."""
    if settings.flask_env != "production":
        return
    if settings.secret_key.strip().lower() in _INSECURE_SECRETS:
        raise RuntimeError(
            "SECRET_KEY is still the shipped development default. "
            "Set a unique SECRET_KEY before starting in production."
        )


def _register_middleware(app: Flask) -> None:
    @app.before_request
    def inject_request_id():
        raw = request.headers.get("X-Request-ID")
        if raw:
            rid = _REQUEST_ID_RE.sub("", raw)[:_MAX_REQUEST_ID_LEN]
        else:
            rid = uuid.uuid4().hex[:12]
        request_id_var.set(rid)
        setattr(request, "request_id", rid)

    @app.after_request
    def harden_response(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    @app.after_request
    def log_response(response):
        logger.info("HTTP %s %s -> %d", request.method, request.path, response.status_code)
        return response


def _register_frontend(app: Flask) -> None:
    def _page(name: str):
        return lambda: render_template(name)  # noqa: E731

    app.add_url_rule("/", view_func=_page("index.html"), endpoint="page_index")
    app.add_url_rule("/evaluate", view_func=_page("evaluate.html"), endpoint="page_evaluate")
    app.add_url_rule("/result", view_func=_page("result.html"), endpoint="page_result")
    app.add_url_rule("/dashboard", view_func=_page("dashboard.html"), endpoint="page_dashboard")


def _build_services(app: Flask) -> None:
    from app.services.evaluation_service import EvaluationService
    from app.services.ocr_service import get_ocr_service

    settings = app.config["SETTINGS"]
    ocr_service = get_ocr_service(settings)
    app.config["OCR_SERVICE"] = ocr_service
    app.config["EVALUATION_SERVICE"] = EvaluationService(
        settings=settings, ocr_service=ocr_service
    )


def create_app(config: dict | None = None, settings=None) -> Flask:
    """Build the Flask app.

    `settings` may be supplied to bypass the environment entirely (tests do
    this). A `SETTINGS` entry in `config` is honoured too, but it only replaces
    the settings object - the derived keys (database URI, upload limit, ...) are
    re-derived from it, so an override can never leave a stale
    `SQLALCHEMY_DATABASE_URI` pointing somewhere else.
    """
    settings = settings or load_settings()
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config.update(flask_config(settings))
    if config:
        app.config.update(config)
    settings = app.config["SETTINGS"]
    app.config.update(
        {key: value for key, value in flask_config(settings).items() if key != "SETTINGS"}
    )
    app.json.sort_keys = False

    _guard_production_secrets(settings)

    setup_logging(settings.log_path, settings.log_level)

    db.init_app(app)
    migrate.init_app(app, db)
    with app.app_context():
        # Importing the package registers every table on db.metadata.
        importlib.import_module("app.models")

        if settings.flask_env not in ("production",):
            db.create_all()

    CORS(app, resources={r"/api/v1/*": {"origins": settings.cors_origins}})

    register_error_handlers(app)
    _register_middleware(app)
    _build_services(app)
    _register_frontend(app)

    from app.routes.evaluation import bp as evaluation_bp
    from app.routes.health import bp as health_bp
    from app.routes.ocr import bp as ocr_bp
    from app.routes.student import bp as student_bp

    prefix = "/api/v1"
    app.register_blueprint(health_bp, url_prefix=prefix)
    app.register_blueprint(evaluation_bp, url_prefix=prefix)
    app.register_blueprint(ocr_bp, url_prefix=prefix)
    app.register_blueprint(student_bp, url_prefix=prefix)

    # Templates must not hardcode limits the server enforces.
    @app.context_processor
    def inject_limits():
        return {"limits": public_limits(settings)}

    from src.ml.model import model_manager

    # Always hand the manager the configured settings, even when preload is
    # off. The manager then loads lazily on the first evaluation, and
    # model_info() can report the configured mode. Without this, /model/info
    # returned model_mode="unknown" and the home page mislabelled a development
    # deployment as "Production mode".
    model_manager.configure(settings)

    if settings.model_preload:
        model_manager.load(settings)
        logger.info("Model preloaded | mode=%s", settings.model_mode)

    logger.info("Application startup complete | env=%s mode=%s", settings.flask_env, settings.model_mode)
    return app