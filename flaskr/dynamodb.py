import boto3
import json

class DynamoDBWrapper:
    def __init__(self, table_name):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table(table_name)

    def put_item(self, item: dict[str: any]) -> str:
        """
        item: dict()
        { timestamp: string,
          input_text: string,
          output_text: string,
          temperature: number,
          top_p: number
        }
        """
        try:
            response = self.table.put_item(Item=item)
            return response
        except Exception as e:
            print(f"Error putting item: {e}")
            raise

    def get_item(self, key):
        response = self.table.get_item(Key=key)
        return response.get('Item')