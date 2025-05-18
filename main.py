from app import app  # noqa: F401

# This file is needed to run the application
# It imports the app from app.py

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
