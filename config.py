class Config:
    SECRET_KEY = 'your_secret_key'
    DEBUG = False
    TESTING = False
    # Add other common configurations here

class DevelopmentConfig(Config):
    DEBUG = True
    # Add development-specific configurations here

class TestingConfig(Config):
    TESTING = True
    # Add testing-specific configurations here

class ProductionConfig(Config):
    # Add production-specific configurations here
    pass

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}