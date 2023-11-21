from flask import Flask
from views import main_views
from sqlalchemy import create_engine, text

app = Flask(__name__)

app.config.from_pyfile('config.py')

database_url_with_charset = app.config['DB_URL']
database = create_engine(database_url_with_charset)
app.database = database

app.register_blueprint(main_views.bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3123, debug=True)