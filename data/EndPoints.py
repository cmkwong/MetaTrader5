# connecting NodeJS server (request API endpoints)
class EndPoints:
    mainUrl = "http://localhost:3002/"
    uploadForexDataUrl = mainUrl + "api/v1/query/forexTable/upload?tableName={}"
    downloadForexDataUrl = mainUrl + "api/v1/query/forexTable/download?tableName={}"
    createTableUrl = mainUrl + "api/v1/query/forexTable/create?tableName={}"
