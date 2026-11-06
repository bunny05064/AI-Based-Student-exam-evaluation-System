"""Fail if the models have drifted from the migration history.

Why this exists instead of `flask db migrate --check`
----------------------------------------------------
Flask-Migrate has no `--check` flag (verified against 4.1.0: "No such option
'--check'"). Alembic ships the equivalent as `alembic check`, but invoking it
needs an `alembic.ini`, which this project does not have -- it configures
Alembic programmatically through Flask-Migrate.

So this script calls the underlying comparison directly: it diffs the live
SQLAlchemy metadata against the schema in the database. An empty diff means a
`flask db migrate` would generate nothing, i.e. every model change is already
covered by a revision.

Exit codes:
    0 - no drift
    1 - drift (a migration is missing)
    2 - setup problem (no database, models missing, etc.)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _brief(obj: object) -> str:
    """Best short label for a diff payload object (Column, Index, Constraint...)."""
    name = getattr(obj, "name", None)
    if name:
        return str(name)
    index = getattr(obj, "index", None)
    if index is not None:
        columns = ", ".join(str(getattr(c, "name", c)) for c in getattr(index, "columns", []))
        return f"index on ({columns})" if columns else "index"
    return obj.__class__.__name__


def _describe(item: object) -> str:
    """Render one compare_metadata diff entry as readable text.

    Alembic's shape changed across versions: 1.18 yields
    ``(operation, object_name, table_name, payload)`` while older releases yield
    2- or 3-tuples, and some paths use dicts. All of them are handled so this
    script does not depend on one Alembic release.
    """
    if isinstance(item, dict):
        target = item.get("table_name") or item.get("name") or ""
        return f"{item.get('type', '?')} on {target}".strip()

    if isinstance(item, tuple) and item:
        kind = str(item[0])
        payload = tuple(item[3:]) if len(item) >= 4 else tuple(item[2:])

        # item[2] is the table name in the 4-tuple form; in the 3-tuple form the
        # only useful thing at index 2 is the payload itself.
        table = str(item[2]) if len(item) >= 4 and isinstance(item[2], str) else None

        parts = [_brief(p) for p in payload if p is not None]
        if table:
            parts.insert(0, table)
        return f"{kind}: {' -> '.join(parts)}" if parts else kind

    return str(item)


def main() -> int:
    try:
        from alembic.autogenerate import compare_metadata
        from alembic.runtime.migration import MigrationContext
    except ImportError as exc:  # noqa: BLE001
        print(f"Alembic is not importable: {exc}", file=sys.stderr)
        return 2

    try:
        from app import create_app
        from app.extensions import db
    except Exception as exc:  # noqa: BLE001
        print(f"Could not import the application: {exc}", file=sys.stderr)
        return 2

    try:
        app = create_app()
    except Exception as exc:  # noqa: BLE001
        print(f"Could not create the app: {exc}", file=sys.stderr)
        return 2

    with app.app_context():
        if not db.metadata.tables:
            print("No models are registered on db.metadata.", file=sys.stderr)
            return 2

        try:
            # engine.connect() rather than raw_connection(): SQLAlchemy 2.x
            # returns a PoolProxiedConnection from raw_connection(), which does
            # not support the context-manager protocol.
            with db.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                diff = compare_metadata(context, db.metadata)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not compare schema against '{app.config.get('SQLALCHEMY_DATABASE_URI')}': {exc}")
            return 2

    if diff:
        print("Model/schema drift detected. The models contain changes with no migration.")
        for item in diff:
            print(f"  - {_describe(item)}")
        print("\nRun:  flask db migrate -m \"<describe the change>\"")
        print("Then: flask db upgrade")
        return 1

    print("No drift: models match the migration history.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
