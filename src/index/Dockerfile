FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONPATH "${PYTHONPATH}:${LAMBDA_TASK_ROOT}/lib:${LAMBDA_TASK_ROOT}/lib/python:${LAMBDA_TASK_ROOT}/lib/python3.11"

RUN curl -O https://lambda-insights-extension.s3-ap-northeast-1.amazonaws.com/amazon_linux/lambda-insights-extension.rpm && \
    rpm -U lambda-insights-extension.rpm && \
    rm -f lambda-insights-extension.rpm
    
# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY index/requirements-frozen.txt  requirements-frozen.txt
RUN  pip3 install -r requirements-frozen.txt --target "${LAMBDA_TASK_ROOT}"

COPY shared ./shared/
RUN pip3 install ./shared --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY index/__init__.py ${LAMBDA_TASK_ROOT}
COPY index/index_file ${LAMBDA_TASK_ROOT}/index_file
COPY index/app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]