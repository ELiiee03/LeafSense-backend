# from app import create_app

# app = create_app()

# if __name__ == "__main__":
#     app.run(debug=True)

# from app import create_app

# app = create_app()

# if __name__ == "__main__":
#     app.run(
#         debug=True, 
#         host='0.0.0.0',
#         port=5000,
#         use_reloader=False,
#         threaded=True
#     )

from app import create_app

app = create_app()

if __name__ == "__main__":
    from pyngrok import ngrok
    
    # Open a ngrok tunnel to your Flask app
    port = 5000
    public_url = ngrok.connect(port).public_url
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
    
    # Update any base URLs or webhooks with the public ngrok URL
    app.config["BASE_URL"] = public_url
    
    # Start the Flask app
    app.run(
        debug=True, 
        host='0.0.0.0',
        port=port,
        use_reloader=False,
        threaded=True
    )