# from app import create_app

# app = create_app()

# if __name__ == "__main__":
#     app.run(debug=True)

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        debug=True, 
        host='0.0.0.0',
        port=5000,
        # ssl_context=('cert.pem', 'key.pem'),
        # Add these options for development
        use_reloader=False,
        threaded=True
    )