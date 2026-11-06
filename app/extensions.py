"""Shared application extensions (declared once, initialized in create_app)."""

from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine

db = SQLAlchemy()
migrate = Migrate()


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    """SQLite ignores FK constraints unless asked; Postgres enforces them.

    Without this, dev/test runs happily store orphan rows that a Postgres
    production deployment would reject with an IntegrityError.
    """
    if type(dbapi_connection).__module__.startswith("sqlite3"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()