import requests

# connecting NodeJS server (request API endpoints)
class EndPoints:
    def __init__(self):
        self.mainUrl = "http://localhost:3002/"
        self.uploadForexData = "api/v1/query/uploadForex?tableName={}"
        self.getForexData = "api/v1/query/getForex?tableName={}"
        self.createTable = "api/v1/query/createTable?tableName={}"

# upload the forex loader
class DataController(EndPoints):

    def postForexData(self, tableName:str, data:list):
        """
        loader: [dist]
        """
        print(f"Data being uploaded: {len(data)}")
        result = requests.post(self.mainUrl + self.uploadForexData.format(tableName), json=data)
        if result.status_code != 200:
            print("Failed to loaded. ")
            return False
        return True

    def getForexData(self, tableName):
        pass

    def createTable(self, tableName, colDict):
        pass