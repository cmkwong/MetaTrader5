# connecting NodeJS server (request API endpoints)
class EndPoints:
    mainUrl = "http://localhost:3002/"
    uploadForexDataUrl = mainUrl + "api/v1/query/forex?tableName={}"
    downloadForexDataUrl = mainUrl + "api/v1/query/forex?tableName={}"
    createTableUrl = mainUrl + "api/v1/query/createForexTable?tableName={}"
