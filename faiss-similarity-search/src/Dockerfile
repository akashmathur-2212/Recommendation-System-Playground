FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy function code
COPY . . 

# Set the CMD to your handler
CMD ["lambda_handler.lambda_handler"]