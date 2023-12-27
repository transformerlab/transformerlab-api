import os
import boto3

# check if environment variables are set
if not os.environ.get('AWS_ACCESS_KEY_ID'):
    raise ValueError("No AWS_ACCESS_KEY_ID set for upload_to_s3.py")

# get environment variable for AWS id and key:
AWS_ACCESS_KEY_ID: str | None = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY: str | None = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_BUCKET_NAME: str | None = os.environ.get('AWS_BUCKET_NAME')

# connect to s3
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# delete old zip file from filesystem:
os.system(command='rm llmlab_api.zip')

# zip all files in this project
os.system(command='git archive -o llmlab_api.zip HEAD')

# get current version of llmlab
with open(file='VERSION', mode='r') as f:
    version: str = f.read().strip()

# upload zip file to s3
s3.upload_file('llmlab_api.zip', AWS_BUCKET_NAME,
               'llmlab_api_v' + version + '.zip')
