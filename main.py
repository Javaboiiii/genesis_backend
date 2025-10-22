import __init__  
import os 
from fastapi import FastAPI, Request, Response
import uvicorn
import json 
from pydantic import BaseModel
from vector_db import pc 
from google import genai

app = FastAPI() 

index = pc.Index(os.getenv('INDEX_NAME'))

client = genai.Client()

@app.route('/embed_data', methods=['POST'])
async def insert_data(req: Request): 
    try:
        body = await req.json()
        index.upsert(namespace=body['namespace'], vectors=[body['vector']]) 
        res = {
            "message": "Data embeded successfully"
        }

        return Response(json.dumps(res), status_code=200) 
    except:
        response = {
            "message": "Unable to embedd data. May be some field is missing"
        }
        return Response(json.dump(response), status_code=400)

@app.route("/report", methods=['GET'])
async def getReport(req: Request) : 
    try:
        data = await req.json()

        print(data['namespace'])

        result = index.search(
            namespace=data['namespace'], 
            query={
                "top_k": 5,
                "inputs":{
                    "text": data['query'] 
                }

            }
        )

        ai_res = client.models.generate_content(
            model="gemini-2.5-flash", contents=f"Give me neat, short response on my vector database result. Below is the query ${data['query']} by user ${data['namespace']} and below is result from my pinecone database ${result}. Now you need to do is give a neat result based on query and result to the user. Also don't let him know the we are using the database and ai is generating this result. I want neat and clean humanise answer"
        )

        response = {
            "result": ai_res.text
        }

        return Response(json.dumps(response), status_code=200, headers={
            "Content-Type": "application/json"
        })
    except: 
        response = {
            "message": "Unable to query. Try again"
        }
        return Response(json.dumps(response), status_code=400, headers={
            "Content-Type": "application/json"
        })



if __name__ == "__main__" : 
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 