from flaskr import app
from flask import render_template, request, redirect, url_for
from flaskr.bedrock import BedrockLLMWrapper
from flaskr.dynamodb import DynamoDBWrapper
import datetime

MAX_TOKEN = 1000
HISTORY_QUERY_LIMIT = 5
DYNAMODB_TABLE_NAME = 'bedrock-poc-flaskr'

@app.route('/')
def index():
    return render_template(
        'index.html',
    )

@app.route('/history')
def history():
    if request.args.get('limit') is not None:
        limit = int(request.args.get('limit'))
    else:
        limit = HISTORY_QUERY_LIMIT
    dynamodb = DynamoDBWrapper(table_name=DYNAMODB_TABLE_NAME)
    items = dynamodb.table.scan(Limit=limit)['Items']
    return render_template(
        'history.html',
        items=items
    )

@app.route('/invoke_model', methods=['POST'])
def invoke_model():
    input_text = request.form['input_text']
    temperature = float(request.form['temperature'])
    top_p = float(request.form['top_p'])
    model_id = request.form['modelId']

    bedrock = BedrockLLMWrapper(model_id,region_name='ap-northeast-1')
    output_text = bedrock.invoke(input_text, temperature, top_p)

    dt = datetime.datetime.now()
    timestamp = dt.strftime('%Y%m%dT%H%M%S')

    dynamodb = DynamoDBWrapper(table_name=DYNAMODB_TABLE_NAME)
    dynamodb.put_item({
        'timestamp': timestamp,
        'input_text': input_text,
        'output_text': output_text,
        'model_id': model_id,
        'temperature': str(temperature),
        'top_p': str(top_p)
    })

    return render_template(
        'index.html',
        output_text=output_text
        )