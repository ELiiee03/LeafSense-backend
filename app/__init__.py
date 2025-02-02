# from flask import Flask
# from config import config

# def create_app(config_name='default'):
#     app = Flask(__name__)
#     app.config.from_object(config[config_name])

#     # Import and register blueprints
#     from .routes import main as main_blueprint
#     app.register_blueprint(main_blueprint)

#     return app

from flask import Flask
from flask_cors import CORS
from .routes import main

def create_app():
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:8100", "capacitor://localhost", "http://localhost", "http://192.168.1.88" ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Register blueprint
    app.register_blueprint(main)
    
    return app