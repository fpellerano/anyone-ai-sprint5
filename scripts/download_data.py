import boto3
import os
import dotenv

dotenv.load_dotenv()

aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY')

print(aws_access_key_id)
print(aws_secret_access_key)
