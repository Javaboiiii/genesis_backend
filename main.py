import __init__  
import os 
from fastapi import FastAPI, Request, Response
import uvicorn
import json 
from pydantic import BaseModel
from vector_db import pc 

app = FastAPI() 


@app.route('/embed_data', methods=['POST'])
async def insert_data(req: Request): 
    # try:
        body = await req.json()
        index = pc.Index(os.getenv('INDEX_NAME'))
        index.upsert(namespace=body['namespace'], vectors=[body['vector']]) 
        res = {
            "message": "Data embeded successfully"
        }

        return Response(json.dumps(res), status_code=200) 
    # except: 

    #     res = {
    #         "message": "Failed to embed data"
    #     }
    #     return Response(json.dumps(res), status_code=500) 

@app.route("/report", methods=['GET'])
async def getReport(req: Request) : 
    data = await req.json()

    return Response(content=json.dumps(data), status_code=200, headers={
        "Content-Type": "application/json"
    })


if __name__ == "__main__" : 
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 