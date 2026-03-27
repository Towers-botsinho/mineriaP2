# Default entry point for AWS Elastic Beanstalk
from app.app import app as application

if __name__ == "__main__":
    application.run()
