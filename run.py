# Root entrypoint.
"""Run the Flask application:  python run.py [PORT]

For production use gunicorn (POSIX):  gunicorn --bind 0.0.0.0:8000 run:app
"""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = app.config.get("FLASK_DEBUG", False)
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)